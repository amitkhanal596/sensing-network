/*
A minimal Web Audio stub for testing scheduling logic in Node.

This does not synthesise audio. It records what the engine *asks* the audio
system to do -- which oscillators were created, at what frequency, and when
each was started and stopped -- so the properties that actually broke in
practice (marker and data tone overlapping, wrong node routing, voices never
stopped) can be asserted in CI. Real signal verification still needs a browser;
this covers the control logic that a browser check is bad at inspecting.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

class Param {
  constructor(value = 0, log = null, name = "") {
    this.value = value;
    this._log = log;
    this._name = name;
    this.events = [];
  }
  _record(kind, value, time) {
    this.events.push({ kind, value, time });
    if (this._log) this._log.push({ param: this._name, kind, value, time });
    return this;
  }
  setValueAtTime(v, t) { this.value = v; return this._record("set", v, t); }
  linearRampToValueAtTime(v, t) { this.value = v; return this._record("linear", v, t); }
  exponentialRampToValueAtTime(v, t) { this.value = v; return this._record("exp", v, t); }
  setTargetAtTime(v, t) { this.value = v; return this._record("target", v, t); }
  cancelScheduledValues(t) { return this._record("cancel", null, t); }
}

class Node {
  constructor(ctx, type) {
    this.ctx = ctx;
    this.type_ = type;
    this.connections = [];
  }
  connect(target) { this.connections.push(target); return target; }
  disconnect() { this.connections.length = 0; }
}

class Oscillator extends Node {
  constructor(ctx) {
    super(ctx, "oscillator");
    this.type = "sine";
    this.frequency = new Param(440);
    this.detune = new Param(0);
    this.startTime = null;
    this.stopTime = null;
    ctx.oscillators.push(this);
  }
  start(t = this.ctx.currentTime) {
    if (this.startTime !== null) throw new Error("oscillator started twice");
    this.startTime = t;
  }
  stop(t = this.ctx.currentTime) { this.stopTime = t; }
}

class Gain extends Node {
  constructor(ctx) {
    super(ctx, "gain");
    this.gain = new Param(1, ctx.paramLog, "gain");
    ctx.gains.push(this);
  }
}

class BiquadFilter extends Node {
  constructor(ctx) {
    super(ctx, "filter");
    this.type = "lowpass";
    this.frequency = new Param(350);
    this.Q = new Param(1);
    ctx.filters.push(this);
  }
}

class StereoPanner extends Node {
  constructor(ctx) {
    super(ctx, "panner");
    this.pan = new Param(0);
    ctx.panners.push(this);
  }
}

class Compressor extends Node {
  constructor(ctx) {
    super(ctx, "compressor");
    this.threshold = new Param(-24);
    this.knee = new Param(30);
    this.ratio = new Param(12);
    this.attack = new Param(0.003);
    this.release = new Param(0.25);
  }
}

export class StubAudioContext {
  constructor({ maxChannelCount = 2, sampleRate = 48000 } = {}) {
    this.state = "running";
    this.sampleRate = sampleRate;
    this.currentTime = 0;
    this.oscillators = [];
    this.gains = [];
    this.filters = [];
    this.panners = [];
    this.paramLog = [];
    this.destination = { maxChannelCount, connect() {}, };
  }
  createOscillator() { return new Oscillator(this); }
  createGain() { return new Gain(this); }
  createBiquadFilter() { return new BiquadFilter(this); }
  createStereoPanner() { return new StereoPanner(this); }
  createDynamicsCompressor() { return new Compressor(this); }
  createAnalyser() {
    return { fftSize: 2048, frequencyBinCount: 1024, connect() {}, };
  }
  async resume() { this.state = "running"; }
  advance(seconds) { this.currentTime += seconds; }
  reset() {
    this.oscillators.length = 0;
    this.gains.length = 0;
    this.filters.length = 0;
    this.panners.length = 0;
    this.paramLog.length = 0;
  }
}

/**
 * Install the stub as the global AudioContext. Returns the context so a test
 * can inspect what the engine scheduled.
 */
export function installStubAudio(options = {}) {
  const ctx = new StubAudioContext(options);
  globalThis.window = globalThis.window || {};
  globalThis.window.AudioContext = function () { return ctx; };
  globalThis.window.webkitAudioContext = globalThis.window.AudioContext;
  return ctx;
}
