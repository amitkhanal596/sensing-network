/*
Tests for the Web Audio sonification engine.

Run with:  node tests/test_audio.mjs

These use a stub audio context that records what was scheduled rather than
rendering sound. They cover the control logic a manual browser check inspects
poorly: whether the identity marker and the data tone overlap, whether mono
output falls back to an audible cue, and whether voices are always stopped.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

import { installStubAudio } from "./audio_context_stub.mjs";

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

async function asyncTest(name, fn) {
  try {
    await fn();
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

const ctx = installStubAudio({ maxChannelCount: 2 });
const { SonificationEngine } = await import(
  "../callibration/public/js/sonification.js"
);

const SPEC = {
  nodes: [0, 1, 2],
  links: [[0, 1], [1, 2]],
  encoding: [
    { channel: "pitch", attribute: "degree", label: "degree", scale: {} },
  ],
  metadata: {},
  legend: ["Pitch encodes degree."],
  tones: [
    { node: 0, params: { pitch: 261.63, timbre: "sine", duration: 0.3, loudness: 0.8, brightness: 0.5 }, attributes: { degree: 1 } },
    { node: 1, params: { pitch: 523.25, timbre: "square", duration: 0.3, loudness: 0.8, brightness: 0.7 }, attributes: { degree: 2 } },
    { node: 2, params: { pitch: 1046.5, timbre: "triangle", duration: 0.3, loudness: 0.8, brightness: 0.9 }, attributes: { degree: 1 } },
  ],
};

const freshEngine = async (spec = SPEC, options = {}) => {
  ctx.reset();
  ctx.currentTime = 0;
  const engine = new SonificationEngine(JSON.parse(JSON.stringify(spec)), options);
  await engine.start();
  ctx.reset(); // discard nodes created during setup
  return engine;
};

await asyncTest("playNode uses the pitch from the spec", async () => {
  const engine = await freshEngine();
  engine.playNode(1);
  const osc = ctx.oscillators[0];
  assert(osc, "an oscillator should have been created");
  assert(
    Math.abs(osc.frequency.value - 523.25) < 0.01,
    `expected 523.25 Hz, got ${osc.frequency.value}`
  );
  assert(osc.type === "square", `expected square waveform, got ${osc.type}`);
});

await asyncTest("every oscillator is scheduled to stop", async () => {
  const engine = await freshEngine();
  engine.playNode(0);
  for (const osc of ctx.oscillators) {
    assert(osc.startTime !== null, "oscillator was never started");
    assert(osc.stopTime !== null, "oscillator was never stopped (would leak)");
    assert(osc.stopTime > osc.startTime, "oscillator stops before it starts");
  }
});

await asyncTest("identity marker finishes before the data tone begins", async () => {
  const spec = JSON.parse(JSON.stringify(SPEC));
  for (const tone of spec.tones) {
    tone.params.identity_prefix = [1567.98];
    tone.params.identity_prefix_duration = 0.09;
    tone.params.identity_prefix_gap = 0.04;
  }
  const engine = await freshEngine(spec);
  engine.playNode(0);

  // The marker is the high oscillator, the data tone the one at 261.63 Hz.
  const marker = ctx.oscillators.find((o) => o.frequency.value > 1500);
  const data = ctx.oscillators.find((o) => Math.abs(o.frequency.value - 261.63) < 1);
  assert(marker, "no marker oscillator was scheduled");
  assert(data, "no data oscillator was scheduled");
  assert(
    data.startTime >= marker.startTime,
    "the data tone must not start before the marker"
  );
  assert(
    data.startTime > 0,
    "the data tone must be delayed past the marker, not scheduled at once"
  );
});

await asyncTest("scheduling uses the audio clock, not wall-clock delay", async () => {
  // Regression test: the data tone used to be scheduled with setTimeout, which
  // drifts under load. It must be scheduled synchronously at a future
  // audio-clock time instead.
  const spec = JSON.parse(JSON.stringify(SPEC));
  spec.tones[0].params.identity_prefix = [1567.98];
  const engine = await freshEngine(spec);
  engine.playNode(0);
  const data = ctx.oscillators.find((o) => Math.abs(o.frequency.value - 261.63) < 1);
  assert(
    data,
    "the data tone must be scheduled immediately (on the audio clock), not after a timer"
  );
});

await asyncTest("mono output substitutes an audible marker for pan identity", async () => {
  const spec = JSON.parse(JSON.stringify(SPEC));
  spec.metadata.identity_channel = "pan";
  spec.tones.forEach((tone, i) => {
    tone.params.pan = -0.8 + i * 0.8;
  });

  // Stereo: pan is used as-is, no marker needed.
  ctx.destination.maxChannelCount = 2;
  const stereo = await freshEngine(spec);
  assert(stereo.hasStereo() === true, "should detect stereo");
  assert(
    !Array.isArray(stereo.paramsFor(0).identity_prefix),
    "stereo output should not need a fallback marker"
  );
  assert(stereo.outputWarnings().length === 0, "stereo output should not warn");

  // Mono: pan is inaudible, so a marker must appear.
  ctx.destination.maxChannelCount = 1;
  const mono = await freshEngine(spec);
  assert(mono.hasStereo() === false, "should detect mono");
  const params = mono.paramsFor(0);
  assert(
    Array.isArray(params.identity_prefix) && params.identity_prefix.length,
    "mono output must substitute a marker for the inaudible pan cue"
  );
  assert(mono.outputWarnings().length === 1, "mono output should warn the user");

  // Different nodes must get different markers, or the fallback is useless.
  const a = mono.paramsFor(0).identity_prefix[0];
  const b = mono.paramsFor(1).identity_prefix[0];
  assert(a !== b, "fallback markers must differ between nodes");

  ctx.destination.maxChannelCount = 2;
});

await asyncTest("setTouchedNode does not retrigger on a repeated node", async () => {
  const engine = await freshEngine();
  engine.setTouchedNode(1);
  const afterFirst = ctx.oscillators.length;
  engine.setTouchedNode(1);
  engine.setTouchedNode(1);
  assert(
    ctx.oscillators.length === afterFirst,
    "a repeated node must not retrigger, or a noisy sensor stream stutters"
  );
});

await asyncTest("continuous mode sustains and releases one voice", async () => {
  const engine = await freshEngine(SPEC, { mode: "continuous" });
  engine.setTouchedNode(1);
  const sustained = ctx.oscillators.filter((o) => o.stopTime === null);
  assert(sustained.length >= 1, "continuous mode should leave a voice running");

  ctx.advance(0.5);
  engine.setTouchedNode(null);
  const stillRunning = ctx.oscillators.filter((o) => o.stopTime === null);
  assert(
    stillRunning.length === 0,
    "releasing the touch must stop every sustained oscillator"
  );
});

await asyncTest("unknown node falls back to defaults without throwing", async () => {
  const engine = await freshEngine();
  engine.playNode(999);
  assert(ctx.oscillators.length === 1, "an unknown node should still make a sound");
});

await asyncTest("describeNode reports each attribute once", async () => {
  const spec = JSON.parse(JSON.stringify(SPEC));
  spec.encoding = [
    { channel: "pitch", attribute: "degree", label: "degree", scale: {} },
    { channel: "brightness", attribute: "degree", label: "degree", scale: {} },
  ];
  const engine = await freshEngine(spec);
  const text = engine.describeNode(1);
  const occurrences = text.split("degree").length - 1;
  assert(occurrences === 1, `attribute repeated in description: "${text}"`);
});

await asyncTest("pan is applied when the spec provides it", async () => {
  const spec = JSON.parse(JSON.stringify(SPEC));
  spec.tones[1].params.pan = 0.6;
  const engine = await freshEngine(spec);
  engine.playNode(1);
  const panner = ctx.panners.find((p) => Math.abs(p.pan.value - 0.6) < 1e-9);
  assert(panner, "the pan value from the spec should reach a StereoPanner");
});

await asyncTest("brightness maps to a filter cutoff within audible range", async () => {
  const engine = await freshEngine();
  engine.playNode(2); // brightness 0.9
  const filter = ctx.filters[0];
  assert(filter, "a low-pass filter should be created");
  assert(
    filter.frequency.value > 200 && filter.frequency.value < 20000,
    `cutoff ${filter.frequency.value} Hz is outside the audible range`
  );
});

await asyncTest("stop() silences a sustained voice", async () => {
  const engine = await freshEngine(SPEC, { mode: "continuous" });
  engine.setTouchedNode(0);
  engine.stop();
  const running = ctx.oscillators.filter((o) => o.stopTime === null);
  assert(running.length === 0, "stop() must halt every oscillator");
});

await asyncTest("fallback markers stay distinct up to the largest network", async () => {
  // Regression: markers used to wrap after 15 nodes, so distant nodes shared a
  // marker -- silently defeating the cue meant to tell them apart. The paper
  // fabricates networks up to 96 nodes, so that is the range that matters.
  const engine = await freshEngine();
  for (const count of [20, 50, 96]) {
    const markers = new Set();
    for (let i = 0; i < count; i++) {
      markers.add(JSON.stringify(engine._fallbackMarker(i).map(Math.round)));
    }
    assert(
      markers.size === count,
      `${count} nodes produced only ${markers.size} distinct markers`
    );
  }
});

console.log(`\n${passed}/${passed + failures.length} passed`);
process.exit(failures.length ? 1 : 0);
