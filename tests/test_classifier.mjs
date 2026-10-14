/*
Tests for the touch classifier and stability tracker.

Run with:  node tests/test_classifier.mjs

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

import { TouchClassifier, StableNodeTracker, NO_NODE } from "../callibration/public/js/classifier.js";

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  pass  ${name}`);
  } catch (e) {
    failures.push([name, e]);
    console.log(`  FAIL  ${name}: ${e.message}`);
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

// Deterministic pseudo-random noise, so the suite cannot flake.
function makeRng(seed) {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
}

function syntheticSamples(means, { noise = 20, perNode = 60, settleMs = 1500 } = {}) {
  const rng = makeRng(42);
  const samples = [];
  for (const [node, mu] of Object.entries(means)) {
    let time = 0;
    for (let i = 0; i < perNode; i++) {
      // The first samples are deliberately wrong, standing in for the moment
      // a hand is still moving; fit() should discard them.
      const settling = time < settleMs;
      const value = settling ? mu * 0.4 : mu + (rng() - 0.5) * noise;
      samples.push({ node, time, value });
      time += 50;
    }
  }
  return samples;
}

const MEANS = { "-1": 100, 0: 300, 1: 500, 2: 700, 3: 900 };

test("fit discards unsettled samples", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const p = c.profiles.get("1");
  assert(Math.abs(p.mean - 500) < 15, `mean drifted to ${p.mean}`);
});

test("fit produces one profile per node", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  assert(c.profiles.size === 5, `expected 5 profiles, got ${c.profiles.size}`);
});

test("classifies a clean reading correctly", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  assert(c.classify(700).node === 2, "700 should be node 2");
  assert(c.classify(300).node === 0, "300 should be node 0");
});

test("rejects readings far outside the calibrated range", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const result = c.classify(99999);
  assert(result.node === NO_NODE, "a wild reading must not match a node");
  assert(result.reason === "out-of-range", `unexpected reason ${result.reason}`);
});

test("reports the baseline pseudo-node as no touch", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const result = c.classify(100);
  assert(result.node === NO_NODE, "the baseline must read as no touch");
});

test("confidence is high on a clean reading and lower between nodes", () => {
  const c = new TouchClassifier({ minConfidence: 0 }).fit(syntheticSamples(MEANS));
  const clean = c.classify(700).confidence;
  const between = c.classify(600).confidence;
  assert(clean > 0.9, `clean confidence too low: ${clean}`);
  assert(between < clean, "a reading between nodes must be less confident");
});

test("legacy config format still loads", () => {
  const c = new TouchClassifier().loadConfig({ "-1": 100, 0: 300, 1: 500, 2: 700 });
  assert(c.profiles.size === 4, "legacy config must produce profiles");
  assert(c.classify(500).node === 1, "legacy config must classify");
  assert(c.profiles.get("1").sd > 0, "a spread must be estimated for legacy configs");
});

test("rich config round-trips through toConfig", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const c2 = new TouchClassifier().loadConfig(c.toConfig());
  assert(
    Math.abs(c2.profiles.get("2").mean - c.profiles.get("2").mean) < 1e-9,
    "means must survive a config round trip"
  );
});

test("median filter suppresses isolated spikes", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  let result;
  for (let i = 0; i < 5; i++) result = c.push(700);
  assert(result.node === 2, "steady input should settle on node 2");
  result = c.push(50000); // one wild spike
  assert(result.node === 2, "a single spike must not change the classification");
});

test("separation report flags poorly separated nodes", () => {
  const wide = new TouchClassifier().fit(syntheticSamples(MEANS));
  assert(wide.separationReport().worst.sigmas > 4, "these nodes are well separated");

  const tight = new TouchClassifier().fit(
    syntheticSamples({ 0: 300, 1: 305, 2: 700 }, { noise: 40 })
  );
  assert(
    tight.separationReport().worst.sigmas < 2,
    "nodes 5 units apart with heavy noise must be flagged"
  );
});

test("empty classifier reports no touch rather than throwing", () => {
  const c = new TouchClassifier();
  assert(c.classify(500).node === NO_NODE, "an unfitted classifier must return no touch");
});

test("non-numeric readings are ignored", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const result = c.push("not a number");
  assert(result.node === NO_NODE, "a non-numeric reading must not classify");
});

test("drift tracking shifts profiles with the baseline", () => {
  const c = new TouchClassifier({ driftAlpha: 0.5 }).fit(syntheticSamples(MEANS));
  // Simulate the whole signal drifting upward by 60 units.
  for (let i = 0; i < 40; i++) c.classify(160);
  const result = c.classify(760);
  assert(result.node === 2, `drifted reading misclassified as ${result.node}`);
});

test("tracker requires agreement before committing", () => {
  const tracker = new StableNodeTracker({ enterWindow: 5, enterRatio: 0.7 });
  assert(tracker.push(2).changed === false, "one sample must not commit");
  assert(tracker.push(2).changed === false, "two samples must not commit");
  let result = tracker.push(2);
  result = tracker.push(2);
  assert(result.changed === true, "four of five agreeing must commit");
  assert(result.node === 2, "committed node should be 2");
});

test("tracker ignores single-frame flicker", () => {
  const tracker = new StableNodeTracker({ enterWindow: 5, enterRatio: 0.7 });
  for (let i = 0; i < 5; i++) tracker.push(2);
  const flicker = tracker.push(3);
  assert(flicker.node === 2, "a one-frame blip must not change the node");
  assert(flicker.changed === false, "a one-frame blip must not fire a change");
});

test("tracker commits a genuine node change", () => {
  const tracker = new StableNodeTracker({ enterWindow: 5, enterRatio: 0.7 });
  for (let i = 0; i < 5; i++) tracker.push(2);
  let result;
  for (let i = 0; i < 5; i++) result = tracker.push(3);
  assert(result.node === 3, "a sustained change must commit");
});

test("tracker releases to no touch", () => {
  const tracker = new StableNodeTracker({ enterWindow: 5, enterRatio: 0.7, exitWindow: 4 });
  for (let i = 0; i < 5; i++) tracker.push(2);
  let result;
  for (let i = 0; i < 6; i++) result = tracker.push(NO_NODE);
  assert(result.node === NO_NODE, "releasing must return to no touch");
});

test("full pipeline: samples in, stable node out", () => {
  const c = new TouchClassifier().fit(syntheticSamples(MEANS));
  const tracker = new StableNodeTracker();
  const rng = makeRng(7);
  let committed = null;
  for (let i = 0; i < 20; i++) {
    const reading = 900 + (rng() - 0.5) * 20;
    const { node, changed } = tracker.push(c.push(reading).node);
    if (changed) committed = node;
  }
  assert(committed === 3, `expected node 3, got ${committed}`);
});

// ------------------------------------------------- drift safety (3.1, 3.2)

test("sustained out-of-range readings cannot run the baseline away", () => {
  // Regression: the baseline used to chase any rejected reading, so a hand
  // left on the object or a loose wire would move it past every profile and
  // the device would reject real touches forever.
  const c = new TouchClassifier({ driftAlpha: 0.2 }).loadConfig({
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
    1: { mean: 2200, sd: 25, n: 50 },
  });
  for (let i = 0; i < 50; i++) c.classify(5000);
  const result = c.classify(1600);
  assert(result.node === 0, `a real touch must still classify, got ${result.node}`);
});

test("a finger parked on a node does not move the baseline", () => {
  const c = new TouchClassifier({ driftAlpha: 0.2 }).loadConfig({
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
    1: { mean: 2200, sd: 25, n: 50 },
  });
  for (let i = 0; i < 200; i++) c.classify(1600);
  assert(
    Math.abs(c.baseline - 1000) < 1,
    `baseline drifted to ${c.baseline} while a node was held`
  );
  assert(c.classify(2200).node === 1, "other nodes must still classify");
});

test("the drift correction is bounded", () => {
  const c = new TouchClassifier({ driftAlpha: 0.9 }).loadConfig({
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
    1: { mean: 2200, sd: 25, n: 50 },
  });
  for (let i = 0; i < 500; i++) c.classify(1000 + i);
  assert(
    Math.abs(c._driftOffset()) <= c._maxDrift() + 1e-9,
    "the applied offset must never exceed the cap"
  );
});

test("resetDrift returns to the calibrated baseline", () => {
  const c = new TouchClassifier({ driftAlpha: 0.5 }).loadConfig({
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
  });
  for (let i = 0; i < 20; i++) c.classify(1050);
  c.resetDrift();
  assert(c._driftOffset() === 0, "resetDrift must clear the correction");
});

test("drift model can be identified from labelled touches", () => {
  const cfg = {
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
    1: { mean: 2200, sd: 25, n: 50 },
  };
  const cases = [
    ["multiplicative", (m) => m * 1.15],
    ["additive", (m) => m + 150],
  ];
  for (const [expected, drift] of cases) {
    const c = new TouchClassifier({ driftAlpha: 0.3 }).loadConfig(cfg);
    for (let i = 0; i < 40; i++) c.classify(drift(1000));
    for (const node of ["0", "1"]) {
      for (let i = 0; i < 6; i++) c.observeKnownTouch(node, drift(cfg[node].mean));
    }
    const fit = c.driftModelFit();
    assert(
      fit.model === expected,
      `expected ${expected} drift to be detected, got ${fit.model}`
    );
  }
});

test("applying the detected drift model restores accuracy", () => {
  const cfg = {
    "-1": { mean: 1000, sd: 25, n: 50 },
    0: { mean: 1600, sd: 25, n: 50 },
    1: { mean: 2200, sd: 25, n: 50 },
  };
  const drift = (m) => m * 1.15;
  const c = new TouchClassifier({ driftAlpha: 0.3 }).loadConfig(cfg);
  for (let i = 0; i < 40; i++) c.classify(drift(1000));
  for (const node of ["0", "1"]) {
    for (let i = 0; i < 6; i++) c.observeKnownTouch(node, drift(cfg[node].mean));
  }
  c.driftModel = c.driftModelFit().model;

  let correct = 0;
  let total = 0;
  for (const node of ["0", "1"]) {
    for (let i = 0; i < 20; i++) {
      total += 1;
      if (String(c.classify(drift(cfg[node].mean)).node) === node) correct += 1;
    }
  }
  assert(correct === total, `expected full accuracy, got ${correct}/${total}`);
});

test("the default drift model is the conservative one", () => {
  const c = new TouchClassifier();
  assert(
    c.driftModel === "additive",
    "an unvalidated multiplicative default would overshoot on additive rigs"
  );
});

// --------------------------------------------- distribution shape (3.4)

test("a bimodal node is flagged rather than silently averaged", () => {
  const samples = [];
  let t = 0;
  // Two grip positions produce two clusters; the mean falls in the gap.
  for (let i = 0; i < 100; i++) {
    samples.push({ node: "0", time: t, value: t < 1500 ? 0 : i % 2 ? 300 : 420 });
    t += 50;
  }
  t = 0;
  for (let i = 0; i < 100; i++) {
    samples.push({ node: "1", time: t, value: t < 1500 ? 0 : 360 });
    t += 50;
  }
  const c = new TouchClassifier().fit(samples);
  const report = c.separationReport();
  assert(
    c.profiles.get("0").unimodal === false,
    "the bimodal node should be flagged as non-unimodal"
  );
  assert(
    report.trustworthy === false,
    "the separation report must not claim to be trustworthy here"
  );
  assert(
    report.nonUnimodal.some((d) => d.node === 0),
    "the offending node should be named"
  );
});

test("a well-behaved node is not falsely flagged", () => {
  const rng = makeRng(11);
  const samples = [];
  for (const [node, mu] of Object.entries(MEANS)) {
    let t = 0;
    for (let i = 0; i < 80; i++) {
      samples.push({ node, time: t, value: t < 1500 ? mu * 0.4 : mu + (rng() - 0.5) * 20 });
      t += 50;
    }
  }
  const c = new TouchClassifier().fit(samples);
  const report = c.separationReport();
  assert(
    report.trustworthy === true,
    `clean unimodal data was wrongly flagged: ${JSON.stringify(report.nonUnimodal)}`
  );
});

test("shape information survives a config round trip", () => {
  const samples = [];
  let t = 0;
  for (let i = 0; i < 100; i++) {
    samples.push({ node: "0", time: t, value: t < 1500 ? 0 : i % 2 ? 300 : 420 });
    t += 50;
  }
  const c = new TouchClassifier().fit(samples);
  const restored = new TouchClassifier().loadConfig(c.toConfig());
  assert(
    restored.profiles.get("0").unimodal === false,
    "a warning about data quality must not be lost when saving"
  );
});

console.log(`\n${passed}/${passed + failures.length} passed`);
process.exit(failures.length ? 1 : 0);
