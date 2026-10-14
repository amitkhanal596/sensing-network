/*
Web Audio engine for sensing network physicalizations.

Consumes a sonification spec produced by sensing_network.sonification and
turns a touched-node ID into sound. The engine is deliberately a pure
consumer of the spec: all data-to-sound decisions live in the spec, so the
same file plays any physicalization the Python pipeline produces.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

const DEFAULTS = {
  pitch: 440,
  brightness: 0.6,
  loudness: 0.8,
  duration: 0.35,
  pulse_rate: 0,
  tremolo_depth: 0.5,
  detune: 0,
  pan: 0,
  timbre: "sine",
  attack: 0.01,
};

// Map normalized brightness to a low-pass cutoff. Exponential because pitch
// perception of spectral centroid is roughly logarithmic.
const brightnessToCutoff = (b) => 200 * Math.pow(60, Math.max(0, Math.min(1, b)));

export class SonificationEngine {
  /**
   * @param {object} spec  parsed sonification spec JSON (may be set later)
   * @param {object} options
   *   - masterGain: overall output level (0..1)
   *   - mode: "discrete" (one tone per touch) or "continuous" (sustained
   *     drone while a node stays touched)
   */
  constructor(spec = null, options = {}) {
    this.spec = spec;
    this.ctx = null;
    this.master = null;
    this.limiter = null;
    this.mode = options.mode || "discrete";
    this.masterGain = options.masterGain ?? 0.35;
    this.toneByNode = new Map();
    this.voice = null; // active sustained voice, in continuous mode
    this.currentNode = null;
    this.lastTriggerTime = 0;
    this.minRetriggerInterval = options.minRetriggerInterval ?? 0.08;
    if (spec) this.setSpec(spec);
  }

  setSpec(spec) {
    this.spec = spec;
    this.toneByNode = new Map();
    for (const tone of spec.tones || []) {
      this.toneByNode.set(String(tone.node), tone);
    }
  }

  /**
   * Must be called from a user gesture: browsers block audio otherwise.
   */
  async start() {
    if (!this.ctx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx();

      // A gentle compressor prevents clipping when several channels stack.
      this.limiter = this.ctx.createDynamicsCompressor();
      this.limiter.threshold.value = -8;
      this.limiter.knee.value = 6;
      this.limiter.ratio.value = 8;
      this.limiter.attack.value = 0.003;
      this.limiter.release.value = 0.15;

      this.master = this.ctx.createGain();
      this.master.gain.value = this.masterGain;
      this.master.connect(this.limiter);
      this.limiter.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") await this.ctx.resume();
    return this.ctx.state;
  }

  /**
   * Whether stereo is actually available on this output.
   *
   * This matters because stereo pan is the default identity cue, and pan is
   * silent on a mono speaker or a single earbud -- a very common setup for a
   * blind listener keeping one ear on the room. Reporting false here lets the
   * caller fall back to a cue that survives mono.
   */
  hasStereo() {
    if (!this.ctx) return null;
    if (typeof this.ctx.createStereoPanner !== "function") return false;
    const max = this.ctx.destination.maxChannelCount;
    // maxChannelCount is 0 on some implementations when it cannot be
    // determined; treat only a definite 1 as mono.
    return !(typeof max === "number" && max === 1);
  }

  /**
   * Does this spec depend on a channel the current output cannot reproduce?
   *
   * Returns a list of problems rather than throwing, so the UI can warn and
   * still play.
   */
  outputWarnings() {
    const warnings = [];
    if (!this.spec) return warnings;
    const identityChannel =
      this.spec.metadata && this.spec.metadata.identity_channel;
    if (identityChannel === "pan" && this.hasStereo() === false) {
      warnings.push(
        "This sonification identifies nodes by stereo position, but the " +
          "current audio output is mono, so every node will sound the same. " +
          "Use headphones, or regenerate with identity_strategy='prefix'."
      );
    }
    return warnings;
  }

  setMasterGain(value) {
    this.masterGain = value;
    if (this.master) {
      this.master.gain.setTargetAtTime(value, this.ctx.currentTime, 0.02);
    }
  }

  paramsFor(nodeId) {
    const tone = this.toneByNode.get(String(nodeId));
    const params = { ...DEFAULTS, ...(tone ? tone.params : {}) };

    // If identity rides on stereo pan but the output is mono, the cue is
    // inaudible. Rather than fail silently, fall back to a pitched marker
    // derived from the node's position in the pan ordering, so identity
    // survives on one earbud or a laptop speaker.
    if (
      this.spec &&
      this.spec.metadata &&
      this.spec.metadata.identity_channel === "pan" &&
      !Array.isArray(params.identity_prefix) &&
      this.hasStereo() === false
    ) {
      const index = this.spec.nodes.findIndex(
        (n) => String(n) === String(nodeId)
      );
      if (index >= 0) {
        params.identity_prefix = this._fallbackMarker(index);
        params.identity_prefix_duration = 0.09;
        params.identity_prefix_gap = 0.04;
      }
    }
    return params;
  }

  /**
   * The pitches available for fallback markers: a pentatonic run above the
   * data register, so a marker reads as punctuation rather than measurement.
   */
  _fallbackMarkerPitches() {
    const semitones = [0, 2, 4, 7, 9];
    const pitches = [];
    for (let octave = 0; octave < 3; octave++) {
      for (const step of semitones) {
        pitches.push(440 * Math.pow(2, (72 + step + 12 * octave - 69) / 12));
      }
    }
    return pitches;
  }

  /**
   * A distinct marker for the nth node.
   *
   * One note while the pitches last, then two-note markers, which cover n^2
   * nodes. Wrapping to a single note would make distant nodes collide --
   * silently defeating the cue that exists to tell them apart.
   */
  _fallbackMarker(index) {
    const pitches = this._fallbackMarkerPitches();
    if (index < pitches.length) return [pitches[index]];
    const i = index - pitches.length;
    const first = pitches[Math.floor(i / pitches.length) % pitches.length];
    const second = pitches[i % pitches.length];
    return [first, second];
  }

  /**
   * Build one voice (oscillators + filter + envelope). Returns handles so the
   * caller can sustain or release it.
   */
  _buildVoice(params, { sustained, startTime = null }) {
    const ctx = this.ctx;
    const now = startTime ?? ctx.currentTime;

    const amp = ctx.createGain();
    amp.gain.value = 0;

    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = brightnessToCutoff(params.brightness);
    filter.Q.value = 0.7;

    const panner = ctx.createStereoPanner
      ? ctx.createStereoPanner()
      : null;
    if (panner) panner.pan.value = Math.max(-1, Math.min(1, params.pan));

    const oscillators = [];
    const makeOsc = (detuneCents) => {
      const osc = ctx.createOscillator();
      osc.type = params.timbre;
      osc.frequency.value = params.pitch;
      osc.detune.value = detuneCents;
      osc.connect(filter);
      oscillators.push(osc);
      return osc;
    };
    makeOsc(0);
    // Detune encodes an attribute as "thickness": a second oscillator beats
    // against the first.
    if (params.detune > 0.01) {
      makeOsc(params.detune);
      makeOsc(-params.detune);
    }

    filter.connect(amp);
    if (panner) {
      amp.connect(panner);
      panner.connect(this.master);
    } else {
      amp.connect(this.master);
    }

    // Pulse rate as amplitude modulation: a rhythmic channel, treated in the
    // literature as separable from pitch (unlike loudness or timbre, which
    // interact with it), so it can carry an independent variable.
    let lfo = null;
    let lfoGain = null;
    if (params.pulse_rate > 0.05 && params.tremolo_depth > 0.01) {
      lfo = ctx.createOscillator();
      lfo.type = "sine";
      lfo.frequency.value = params.pulse_rate;
      lfoGain = ctx.createGain();
      lfoGain.gain.value = params.tremolo_depth * params.loudness * 0.5;
      lfo.connect(lfoGain);
      lfoGain.connect(amp.gain);
      lfo.start(now);
    }

    const attack = Math.max(0.002, params.attack);
    const peak = Math.max(0.0001, params.loudness);
    amp.gain.setValueAtTime(0, now);
    amp.gain.linearRampToValueAtTime(peak, now + attack);

    if (!sustained) {
      const dur = Math.max(0.05, params.duration);
      const release = Math.min(0.25, dur * 0.5);
      amp.gain.setValueAtTime(peak, now + Math.max(attack, dur - release));
      amp.gain.exponentialRampToValueAtTime(0.0001, now + dur);
    }

    for (const osc of oscillators) osc.start(now);

    const stopAt = sustained ? null : now + Math.max(0.05, params.duration) + 0.02;
    if (stopAt !== null) {
      for (const osc of oscillators) osc.stop(stopAt);
      if (lfo) lfo.stop(stopAt);
    }

    return { oscillators, amp, filter, lfo, params };
  }

  _releaseVoice(voice, releaseTime = 0.08) {
    if (!voice) return;
    const now = this.ctx.currentTime;
    try {
      voice.amp.gain.cancelScheduledValues(now);
      voice.amp.gain.setValueAtTime(Math.max(voice.amp.gain.value, 0.0001), now);
      voice.amp.gain.exponentialRampToValueAtTime(0.0001, now + releaseTime);
    } catch (e) {
      /* the node may already be finished */
    }
    const stopAt = now + releaseTime + 0.02;
    for (const osc of voice.oscillators) {
      try { osc.stop(stopAt); } catch (e) { /* already stopped */ }
    }
    if (voice.lfo) {
      try { voice.lfo.stop(stopAt); } catch (e) { /* already stopped */ }
    }
  }

  /**
   * Play a short identity marker: a quiet, high blip (or pair of blips) that
   * says *which* node this is before the data tone says what it measures.
   * Returns how long the marker lasts, so the data tone can follow it.
   */
  _playIdentityPrefix(params, startTime = null) {
    const notes = params.identity_prefix;
    if (!Array.isArray(notes) || !notes.length) return 0;

    const ctx = this.ctx;
    const noteDuration = params.identity_prefix_duration ?? 0.09;
    const gap = params.identity_prefix_gap ?? 0.04;
    const base = startTime ?? ctx.currentTime;

    notes.forEach((frequency, index) => {
      const start = base + index * (noteDuration + gap * 0.5);
      const osc = ctx.createOscillator();
      osc.type = "sine";
      osc.frequency.value = frequency;

      const amp = ctx.createGain();
      // Deliberately quieter than the data tone: a marker, not a measurement.
      const peak = 0.28 * (params.loudness ?? 0.8);
      amp.gain.setValueAtTime(0, start);
      amp.gain.linearRampToValueAtTime(peak, start + 0.005);
      amp.gain.exponentialRampToValueAtTime(0.0001, start + noteDuration);

      osc.connect(amp);
      amp.connect(this.master);
      osc.start(start);
      osc.stop(start + noteDuration + 0.02);
    });

    return notes.length * (noteDuration + gap * 0.5) + gap;
  }

  /**
   * Play the tone for a node once, preceded by its identity marker if it has
   * one.
   */
  playNode(nodeId, overrides = {}) {
    if (!this.ctx) return null;
    const params = { ...this.paramsFor(nodeId), ...overrides };
    const now = this.ctx.currentTime;
    const prefixDuration = this._playIdentityPrefix(params, now);

    // Schedule the data tone on the audio clock rather than with setTimeout.
    // setTimeout is subject to main-thread jitter, so under load the marker
    // and the data tone can overlap -- the exact blurring the marker exists
    // to avoid. Web Audio's own clock is sample-accurate.
    return this._buildVoice(params, {
      sustained: false,
      startTime: now + prefixDuration,
    });
  }

  /**
   * Report which node is currently touched (or null for none).
   *
   * In continuous mode this sustains a drone that changes as the touch moves.
   * In discrete mode it fires one tone per new node. Repeated calls with the
   * same node do not retrigger, so a noisy sensor stream can be passed
   * straight through.
   */
  setTouchedNode(nodeId, options = {}) {
    if (!this.ctx) return;
    const normalized = nodeId === null || nodeId === undefined || nodeId === -1
      ? null
      : String(nodeId);

    if (normalized === this.currentNode) return;
    this.currentNode = normalized;

    if (this.mode === "continuous") {
      this._releaseVoice(this.voice);
      this.voice = null;
      if (normalized !== null) {
        const params = { ...this.paramsFor(normalized), ...(options.overrides || {}) };
        this.voice = this._buildVoice(params, { sustained: true });
      }
      return;
    }

    if (normalized !== null) {
      const now = this.ctx.currentTime;
      if (now - this.lastTriggerTime < this.minRetriggerInterval) return;
      this.lastTriggerTime = now;
      this.playNode(normalized, options.overrides || {});
    }
  }

  /**
   * Stop all sound immediately.
   */
  stop() {
    this._releaseVoice(this.voice, 0.03);
    this.voice = null;
    this.currentNode = null;
  }

  /**
   * Play every node in sequence: an overview of the whole dataset, which a
   * static physicalization cannot give on its own.
   */
  async playTour(nodeIds = null, { gap = 0.12, onNode = null } = {}) {
    if (!this.ctx) return;
    const ids = nodeIds || (this.spec ? this.spec.nodes.map(String) : []);
    for (const id of ids) {
      const params = this.paramsFor(id);
      if (onNode) onNode(id);
      this.playNode(id);
      const prefix = Array.isArray(params.identity_prefix)
        ? params.identity_prefix.length *
          ((params.identity_prefix_duration ?? 0.09) +
            (params.identity_prefix_gap ?? 0.04) * 0.5)
        : 0;
      const wait = (Math.max(0.05, params.duration) + prefix + gap) * 1000;
      await new Promise((r) => setTimeout(r, wait));
    }
  }

  /**
   * Play a node and then its neighbours, so a listener can hear local
   * structure without tracing it by hand.
   */
  async playNeighbourhood(nodeId, { gap = 0.1, onNode = null } = {}) {
    if (!this.spec) return;
    const id = String(nodeId);
    const neighbours = [];
    for (const [s, t] of this.spec.links || []) {
      if (String(s) === id) neighbours.push(String(t));
      else if (String(t) === id) neighbours.push(String(s));
    }
    await this.playTour([id, ...neighbours], { gap, onNode });
  }

  /**
   * Text description of a node, for speech output.
   */
  describeNode(nodeId) {
    if (!this.spec) return "";
    const tone = this.toneByNode.get(String(nodeId));
    if (!tone) return `Node ${nodeId} is not in this sonification.`;
    const parts = [`Node ${nodeId}.`];
    const seen = new Set();
    for (const ch of this.spec.encoding || []) {
      if (seen.has(ch.attribute)) continue;
      const value = tone.attributes ? tone.attributes[ch.attribute] : undefined;
      if (value === undefined) continue;
      seen.add(ch.attribute);
      const text = Math.abs(value - Math.round(value)) < 1e-9
        ? String(Math.round(value))
        : Number(value).toPrecision(3);
      parts.push(`${ch.label} ${text}.`);
    }
    return parts.join(" ");
  }

  /**
   * Speak a node's description using the browser's speech synthesis.
   */
  speakNode(nodeId, { rate = 1.1 } = {}) {
    if (!("speechSynthesis" in window)) return false;
    const utterance = new SpeechSynthesisUtterance(this.describeNode(nodeId));
    utterance.rate = rate;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    return true;
  }

  legendText() {
    return (this.spec && this.spec.legend) || [];
  }
}

export default SonificationEngine;
