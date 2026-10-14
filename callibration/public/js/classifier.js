/*
Touch classification from capacitive sensor values.

The original calibration flow assigns a reading to whichever node's mean is
nearest. That is a reasonable first pass, but it has three properties that
become audible once sound is attached:

  1. It always returns a node, even for a reading far from every calibrated
     value -- so releasing the object produces a confident wrong answer.
  2. It ignores each node's spread, so a node with a noisy signature steals
     readings from its tighter neighbours.
  3. It ignores baseline drift. Capacitive readings shift with humidity, hand
     contact area, and what the object rests on, which the paper names as
     open future work.

This classifier addresses all three: it models each node as a mean and
standard deviation, rejects readings that fit nothing, reports a confidence,
and can track slow drift from the untouched baseline.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

const NO_NODE = -1;

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

const stdDev = (xs, mu) => {
  if (xs.length < 2) return 0;
  const m = mu ?? mean(xs);
  const variance = xs.reduce((acc, x) => acc + (x - m) * (x - m), 0) / (xs.length - 1);
  return Math.sqrt(variance);
};

const median = (xs) => {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/**
 * Describe how far a node's samples depart from a single Gaussian.
 *
 * Uses the bimodality coefficient, b = (skew^2 + 1) / kurtosis, corrected for
 * sample size. Values above ~0.555 (the value for a uniform distribution)
 * indicate a bimodal or otherwise non-normal spread. This is a screening
 * heuristic, not a hypothesis test: the point is to warn rather than to
 * silently model two clusters as one.
 */
const distributionShape = (xs, mu, sd) => {
  const n = xs.length;
  if (n < 8 || !(sd > 0)) {
    return { bimodality: null, unimodal: true };
  }
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const x of xs) {
    const d = x - mu;
    m2 += d * d;
    m3 += d * d * d;
    m4 += d * d * d * d;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;
  if (!(m2 > 0)) return { bimodality: null, unimodal: true };

  const skew = m3 / Math.pow(m2, 1.5);
  const kurtosis = m4 / (m2 * m2); // non-excess
  if (!(kurtosis > 0)) return { bimodality: null, unimodal: true };

  // Sample-size correction (SAS definition).
  const correction =
    (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
  const b = (skew * skew + 1) / (kurtosis + correction / n);

  return { bimodality: b, unimodal: b <= 0.555 };
};

export class TouchClassifier {
  /**
   * @param {object} options
   *   - minConfidence: below this, report NO_NODE rather than guess
   *   - rejectSigma: readings further than this many (pooled) sigmas from
   *     every node are treated as "not touching"
   *   - driftAlpha: EMA rate for baseline tracking (0 disables)
   *   - medianWindow: samples used for the median pre-filter
   */
  constructor(options = {}) {
    this.profiles = new Map(); // nodeId -> {mean, sd, n}
    this.minConfidence = options.minConfidence ?? 0.55;
    this.rejectSigma = options.rejectSigma ?? 4.0;
    this.driftAlpha = options.driftAlpha ?? 0.0;
    this.medianWindow = options.medianWindow ?? 5;
    this.baseline = null;
    this.calibratedBaseline = null;
    this.window = [];
    this.minSd = options.minSd ?? 1e-6;

    // Drift tracking is a feedback loop, so it needs bounds. Without them a
    // sustained non-baseline signal (a hand left resting on the object, a
    // loose wire) drags the baseline toward itself, and once the baseline has
    // moved past every profile the device rejects real touches forever.
    // Two guards: only learn from readings that still look like the
    // calibrated baseline, and cap the total correction.
    this.driftLearnSigma = options.driftLearnSigma ?? 3.0;
    // Expressed as a fraction of the calibrated spread between the baseline
    // and the furthest node, so it scales with the network rather than
    // assuming particular sensor units.
    this.maxDriftFraction = options.maxDriftFraction ?? 0.5;
    // How a baseline shift is expected to propagate to the other nodes:
    //
    //   "additive"        every node shifts by the same amount.
    //   "multiplicative"  every reading scales by the ratio the baseline
    //                     moved, which is what a change in hand or ambient
    //                     capacitance does to an RC delay proportional to R*C.
    //
    // Which one holds on real hardware is an open question (CRITICAL_REVIEW
    // 3.2), so the default is the conservative one: additive corrections are
    // bounded and degrade gently when wrong, whereas a multiplicative
    // correction applied to an additive rig overshoots badly on distant
    // nodes. Use observeKnownTouch() + driftModelFit() to measure which model
    // this rig actually follows, then set it deliberately.
    this.driftModel = options.driftModel ?? "additive";
  }

  /**
   * Build profiles from raw calibration samples.
   *
   * @param {Array} samples  [{node, time, value}, ...]
   * @param {object} options
   *   - settleMs: discard this many ms after each node change, so the
   *     samples taken while a hand is still moving do not widen the profile
   *   - trim: fraction trimmed from each tail before computing statistics
   */
  fit(samples, { settleMs = 1500, trim = 0.1 } = {}) {
    const byNode = new Map();
    for (const s of samples) {
      const node = String(s.node);
      if (!byNode.has(node)) byNode.set(node, []);
      byNode.get(node).push({ time: Number(s.time), value: Number(s.value) });
    }

    this.profiles = new Map();
    for (const [node, rows] of byNode) {
      rows.sort((a, b) => a.time - b.time);
      if (!rows.length) continue;
      const startTime = rows[0].time;
      let values = rows
        .filter((r) => r.time >= startTime + settleMs)
        .map((r) => r.value)
        .filter((v) => Number.isFinite(v));
      if (values.length < 3) {
        values = rows.map((r) => r.value).filter((v) => Number.isFinite(v));
      }
      if (!values.length) continue;

      // Trim tails: touch data commonly has a few samples from the moment of
      // contact and release that do not represent a steady touch.
      values.sort((a, b) => a - b);
      const cut = Math.floor(values.length * trim);
      const kept = values.length > 2 * cut + 2 ? values.slice(cut, values.length - cut) : values;

      const mu = mean(kept);
      const sd = Math.max(stdDev(kept, mu), this.minSd);
      // Record how well a single Gaussian actually describes this node. A
      // node touched two different ways (two grip positions, one finger vs.
      // two) is bimodal, and its mean then sits in a gap where it never
      // actually reads -- a mean that looks fine and classifies badly.
      const shape = distributionShape(kept, mu, sd);
      this.profiles.set(node, {
        mean: mu,
        sd,
        n: kept.length,
        median: median(kept),
        bimodality: shape.bimodality,
        unimodal: shape.unimodal,
      });
    }

    // The -1 pseudo-node is the untouched baseline, collected before the
    // first node in the original protocol.
    const base = this.profiles.get(String(NO_NODE));
    if (base) {
      this.baseline = base.mean;
      this.calibratedBaseline = base.mean;
    }
    return this;
  }

  /**
   * Load profiles from a saved config.
   *
   * Accepts both the richer {node: {mean, sd}} form and the original
   * {node: meanValue} form, so existing calibration files still work.
   */
  loadConfig(config) {
    this.profiles = new Map();
    const spreads = [];
    for (const [node, value] of Object.entries(config)) {
      if (value && typeof value === "object") {
        const sd = Math.max(Number(value.sd) || 0, this.minSd);
        this.profiles.set(String(node), {
          mean: Number(value.mean),
          sd,
          n: Number(value.n) || 0,
          ...(value.median !== undefined ? { median: Number(value.median) } : {}),
          ...(value.bimodality != null
            ? { bimodality: Number(value.bimodality) }
            : {}),
          ...(value.unimodal !== undefined ? { unimodal: !!value.unimodal } : {}),
        });
        spreads.push(sd);
      } else {
        this.profiles.set(String(node), { mean: Number(value), sd: null, n: 0 });
      }
    }
    // For legacy configs with no spread information, estimate a common sigma
    // from how far apart the node means are: with k nodes spread over a
    // range, adjacent means sit roughly range/(k-1) apart, and a quarter of
    // that gap is a defensible default width.
    const means = [...this.profiles.values()].map((p) => p.mean).sort((a, b) => a - b);
    if (means.length > 1) {
      const gaps = [];
      for (let i = 1; i < means.length; i++) gaps.push(means[i] - means[i - 1]);
      const typicalGap = median(gaps.filter((g) => g > 0)) || 1;
      const fallbackSd = Math.max(typicalGap / 4, this.minSd);
      for (const p of this.profiles.values()) {
        if (p.sd === null) p.sd = fallbackSd;
      }
    } else {
      for (const p of this.profiles.values()) {
        if (p.sd === null) p.sd = 1;
      }
    }

    const base = this.profiles.get(String(NO_NODE));
    if (base) {
      this.baseline = base.mean;
      this.calibratedBaseline = base.mean;
    }
    return this;
  }

  /**
   * Export profiles in the richer config format.
   */
  toConfig() {
    const out = {};
    for (const [node, p] of this.profiles) {
      out[node] = {
        mean: p.mean,
        sd: p.sd,
        n: p.n,
        ...(p.median !== undefined ? { median: p.median } : {}),
        ...(p.bimodality != null ? { bimodality: p.bimodality } : {}),
        ...(p.unimodal !== undefined ? { unimodal: p.unimodal } : {}),
      };
    }
    return out;
  }

  /**
   * Adjust for slow baseline drift by shifting all profiles to match the
   * currently observed untouched level.
   */
  /**
   * The raw (unbounded) baseline shift observed so far.
   */
  _rawDriftOffset() {
    if (!this.driftAlpha || this.baseline === null || this.calibratedBaseline === null) {
      return 0;
    }
    return this.baseline - this.calibratedBaseline;
  }

  /**
   * The largest baseline correction we are willing to believe, in sensor
   * units. Beyond this the more likely explanation is a fault, not drift.
   */
  /**
   * How far a reading may sit from the calibrated baseline and still be
   * treated as baseline drift.
   *
   * Half the distance to the nearest node keeps the baseline from ever
   * learning its way into another node's territory, while still allowing
   * corrections much larger than the sample noise.
   */
  _baselineLearnLimit() {
    if (this.calibratedBaseline === null || !this.profiles.size) return 0;
    let nearest = Infinity;
    for (const [node, p] of this.profiles) {
      if (parseInt(node, 10) === NO_NODE) continue;
      nearest = Math.min(nearest, Math.abs(p.mean - this.calibratedBaseline));
    }
    if (!Number.isFinite(nearest)) {
      // No real nodes to bound against; fall back to the sample spread.
      const base = this.profiles.get(String(NO_NODE));
      return base ? this.driftLearnSigma * Math.max(base.sd, this.minSd) : 0;
    }
    return nearest * 0.5;
  }

  _maxDrift() {
    if (this.calibratedBaseline === null || !this.profiles.size) return 0;
    let furthest = 0;
    for (const p of this.profiles.values()) {
      furthest = Math.max(furthest, Math.abs(p.mean - this.calibratedBaseline));
    }
    return furthest * this.maxDriftFraction;
  }

  /**
   * Baseline correction, clamped so a runaway can never deafen the device.
   */
  _driftOffset() {
    const raw = this._rawDriftOffset();
    const limit = this._maxDrift();
    if (!limit) return 0;
    return Math.max(-limit, Math.min(limit, raw));
  }

  /**
   * Correction for one node's mean.
   *
   * Two models, because the right one depends on physics we have not measured
   * on real hardware (see CRITICAL_REVIEW.md 3.2):
   *
   *   "additive"  - every node shifts by the same amount. Correct if drift is
   *                 an offset added to the measurement.
   *   "auto"      - every reading scales by the same ratio the baseline moved.
   *                 Correct if drift multiplies the RC time constant, which is
   *                 what a change in hand or environmental capacitance does to
   *                 a delay that is proportional to R*C.
   *
   * Under the multiplicative model a baseline that moved from b to b+d implies
   * every reading scaled by (b+d)/b, so a node at mean m should now read
   * m*(b+d)/b, i.e. an offset of m*d/b.
   */
  _offsetForMean(mean) {
    const offset = this._driftOffset();
    if (offset === 0 || this.driftModel === "additive") return offset;
    if (!this.calibratedBaseline) return offset;
    // "auto" is accepted as an alias for the multiplicative model.
    // Scale the observed baseline shift by this node's reading relative to
    // the baseline's own reading.
    return offset * (mean / this.calibratedBaseline);
  }

  /**
   * Reset drift tracking back to the calibrated baseline.
   */
  resetDrift() {
    this.baseline = this.calibratedBaseline;
    this._driftEvidence = { additive: 0, multiplicative: 0, n: 0 };
    return this;
  }

  /**
   * Decide empirically whether drift on this rig is additive or
   * multiplicative, rather than assuming either.
   *
   * Call with readings taken while touching a *known* node after the baseline
   * has moved. Each observation votes for whichever model predicts it better;
   * `driftModelFit()` reports the tally. This is the cheap version of the
   * calibration improvement the original paper lists as future work: it needs
   * no extra hardware, only a few labelled touches.
   */
  observeKnownTouch(nodeId, value) {
    if (!this._driftEvidence) {
      this._driftEvidence = { additive: 0, multiplicative: 0, n: 0 };
    }
    const p = this.profiles.get(String(nodeId));
    if (!p || this.calibratedBaseline === null) return this._driftEvidence;

    const offset = this._driftOffset();
    if (offset === 0) return this._driftEvidence;

    const additivePrediction = p.mean + offset;
    const multiplicativePrediction =
      p.mean + offset * (p.mean / this.calibratedBaseline);

    const additiveError = Math.abs(value - additivePrediction);
    const multiplicativeError = Math.abs(value - multiplicativePrediction);

    if (additiveError < multiplicativeError) this._driftEvidence.additive += 1;
    else if (multiplicativeError < additiveError) {
      this._driftEvidence.multiplicative += 1;
    }
    this._driftEvidence.n += 1;
    return this._driftEvidence;
  }

  /**
   * Which drift model the observations favour, and how strongly.
   */
  driftModelFit() {
    const evidence = this._driftEvidence || {
      additive: 0,
      multiplicative: 0,
      n: 0,
    };
    if (evidence.n < 5) {
      return { model: null, confidence: 0, evidence, reason: "not enough observations" };
    }
    const decided = evidence.additive + evidence.multiplicative;
    if (!decided) {
      return { model: null, confidence: 0, evidence, reason: "no discriminating observations" };
    }
    const model =
      evidence.additive >= evidence.multiplicative
        ? "additive"
        : "multiplicative";
    const winner = Math.max(evidence.additive, evidence.multiplicative);
    return { model, confidence: winner / decided, evidence };
  }

  /**
   * Feed one raw reading; returns a classification for the smoothed signal.
   *
   * @returns {{node: number, confidence: number, value: number, margin: number}}
   */
  push(rawValue) {
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      return { node: NO_NODE, confidence: 0, value: NaN, margin: 0 };
    }
    this.window.push(value);
    if (this.window.length > this.medianWindow) this.window.shift();
    // A median filter removes the isolated spikes typical of capacitive
    // sensing without smearing genuine transitions the way a mean does.
    return this.classify(median(this.window));
  }

  /**
   * Classify one (already smoothed) value.
   */
  classify(value) {
    if (!this.profiles.size) {
      return { node: NO_NODE, confidence: 0, value, margin: 0 };
    }
    // Score each node by a Gaussian likelihood on its own spread, each
    // shifted by its own drift correction.
    const scored = [];
    for (const [node, p] of this.profiles) {
      const z = (value - (p.mean + this._offsetForMean(p.mean))) / p.sd;
      scored.push({ node, z: Math.abs(z), logLik: -0.5 * z * z - Math.log(p.sd) });
    }
    scored.sort((a, b) => b.logLik - a.logLik);

    const best = scored[0];
    const runnerUp = scored[1];

    // Reject readings that fit nothing: with sound attached, silence is a
    // better answer than a confident wrong note. A value stranded between two
    // profiles is reported separately from one outside the calibrated range,
    // because the two call for different fixes: the first means the nodes are
    // too close to separate, the second means drift or a lost connection.
    if (best.z > this.rejectSigma) {
      this._updateBaseline(value);
      const means = [...this.profiles.values()].map(
        (p) => p.mean + this._offsetForMean(p.mean)
      );
      const between = value > Math.min(...means) && value < Math.max(...means);
      return {
        node: NO_NODE,
        confidence: 0,
        value,
        margin: 0,
        reason: between ? "between-nodes" : "out-of-range",
      };
    }

    // Confidence as the softmax posterior of the best node against the rest.
    const maxLog = best.logLik;
    let total = 0;
    for (const s of scored) total += Math.exp(s.logLik - maxLog);
    const confidence = 1 / total;

    const margin = runnerUp ? Math.abs(best.z - runnerUp.z) : Infinity;
    const nodeId = parseInt(best.node, 10);

    if (nodeId === NO_NODE) {
      this._updateBaseline(value);
      return { node: NO_NODE, confidence, value, margin, reason: "baseline" };
    }

    if (confidence < this.minConfidence) {
      return { node: NO_NODE, confidence, value, margin, reason: "ambiguous" };
    }

    return { node: nodeId, confidence, value, margin };
  }

  _updateBaseline(value) {
    if (!this.driftAlpha) return;
    if (this.baseline === null) {
      this.baseline = value;
      return;
    }
    // Learn only from readings that still plausibly *are* the baseline.
    //
    // The threshold is scaled to the gap to the nearest node, not to the
    // momentary sample noise: genuine drift routinely exceeds a few sigma of
    // instantaneous noise (that is what makes it drift), but a reading that
    // has travelled most of the way to another node is far more likely a
    // touch or a fault than a drifting baseline. Learning from those is what
    // creates a runaway.
    if (this.calibratedBaseline !== null) {
      const limit = this._baselineLearnLimit();
      if (limit > 0 && Math.abs(value - this.calibratedBaseline) > limit) {
        return;
      }
    }
    this.baseline =
      this.baseline * (1 - this.driftAlpha) + value * this.driftAlpha;
  }

  /**
   * How separable the calibrated nodes are, as a diagnostic before printing
   * or during calibration.
   *
   * Returns per-adjacent-pair separation in pooled sigmas; pairs below ~2
   * will be confused in practice.
   */
  separationReport() {
    const entries = [...this.profiles.entries()]
      .filter(([node]) => parseInt(node, 10) !== NO_NODE)
      .map(([node, p]) => ({ node: parseInt(node, 10), ...p }))
      .sort((a, b) => a.mean - b.mean);

    const pairs = [];
    for (let i = 1; i < entries.length; i++) {
      const a = entries[i - 1];
      const b = entries[i];
      const pooled = Math.sqrt((a.sd * a.sd + b.sd * b.sd) / 2) || this.minSd;
      pairs.push({
        nodes: [a.node, b.node],
        gap: b.mean - a.mean,
        sigmas: (b.mean - a.mean) / pooled,
      });
    }
    const worst = pairs.length
      ? pairs.reduce((m, p) => (p.sigmas < m.sigmas ? p : m))
      : null;

    // A node whose samples are not unimodal breaks the assumption behind both
    // the classifier and the sigma figures above, so it is reported: those
    // separations would otherwise look reassuring while being wrong.
    const nonUnimodal = entries
      .filter((e) => e.unimodal === false)
      .map((e) => ({ node: e.node, bimodality: e.bimodality }));

    return {
      pairs,
      worst,
      nNodes: entries.length,
      nonUnimodal,
      trustworthy: nonUnimodal.length === 0,
    };
  }
}

export class StableNodeTracker {
  /**
   * Debounce classifications into stable touch events.
   *
   * Sound makes flicker obvious in a way a text label does not: a node that
   * flips between two values for a few frames is a visible nuisance but an
   * audible mess. This requires agreement across a window before committing,
   * and requires a separate, shorter agreement to release.
   */
  constructor({ enterWindow = 5, enterRatio = 0.7, exitWindow = 4 } = {}) {
    this.enterWindow = enterWindow;
    this.enterRatio = enterRatio;
    this.exitWindow = exitWindow;
    this.history = [];
    this.current = NO_NODE;
  }

  push(node) {
    this.history.push(node);
    const maxWindow = Math.max(this.enterWindow, this.exitWindow);
    if (this.history.length > maxWindow) this.history.shift();

    const recentEnter = this.history.slice(-this.enterWindow);
    const counts = new Map();
    for (const n of recentEnter) counts.set(n, (counts.get(n) || 0) + 1);

    let bestNode = NO_NODE;
    let bestCount = 0;
    for (const [n, c] of counts) {
      if (c > bestCount) {
        bestNode = n;
        bestCount = c;
      }
    }

    const agreed = bestCount >= Math.ceil(this.enterWindow * this.enterRatio);

    if (agreed && bestNode !== this.current) {
      // Release needs only the shorter window, so letting go feels immediate.
      if (bestNode === NO_NODE) {
        const recentExit = this.history.slice(-this.exitWindow);
        if (recentExit.every((n) => n === NO_NODE)) {
          this.current = NO_NODE;
          return { node: this.current, changed: true };
        }
        return { node: this.current, changed: false };
      }
      this.current = bestNode;
      return { node: this.current, changed: true };
    }
    return { node: this.current, changed: false };
  }

  reset() {
    this.history = [];
    this.current = NO_NODE;
  }
}

export { NO_NODE };
export default TouchClassifier;
