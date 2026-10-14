/*
Interactive multimodal explorer: touch a physicalized node, hear its data.

Ties together the sensor stream (via socket.io), the touch classifier, and
the Web Audio sonification engine. Also runs without hardware, so the
sonification design can be developed and tested before anything is printed.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

import { SonificationEngine } from "./sonification.js";
import { TouchClassifier, StableNodeTracker, NO_NODE } from "./classifier.js";

const socket = typeof io !== "undefined" ? io() : null;

const engine = new SonificationEngine(null, { mode: "discrete" });
const classifier = new TouchClassifier({ driftAlpha: 0.02 });
const tracker = new StableNodeTracker();

let spec = null;
let running = false;
let speakEnabled = false;
let lastNode = NO_NODE;

const $ = (sel) => document.querySelector(sel);
const setText = (sel, text) => {
  const el = $(sel);
  if (el) el.textContent = text;
};

// Announce to screen readers without stealing focus.
const announce = (text) => {
  const live = $("#liveRegion");
  if (live) live.textContent = text;
};

async function loadSpec(path = "./data/sonification.json") {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`could not load ${path}`);
  spec = await response.json();
  engine.setSpec(spec);
  renderLegend();
  renderNodeButtons();
  return spec;
}

async function loadConfig(path = "./data/config.json") {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`could not load ${path}`);
  const config = await response.json();
  classifier.loadConfig(config);
  renderSeparation();
  return config;
}

function renderLegend() {
  const el = $("#legend");
  if (!el || !spec) return;
  el.innerHTML = "";
  for (const line of spec.legend || []) {
    const li = document.createElement("li");
    li.textContent = line;
    el.appendChild(li);
  }

  // Say plainly when an identity cue was added, so a reader is not left to
  // infer that pan or a leading blip means something about the data.
  const identity = spec.metadata && spec.metadata.identity;
  const note = $("#identityNote");
  if (note) {
    note.textContent =
      identity && identity.action && identity.action !== "none"
        ? `Identity cue added (${identity.action}): ${identity.reason}.`
        : "";
  }
}

function renderSeparation() {
  const el = $("#separation");
  if (!el) return;
  const report = classifier.separationReport();
  if (!report.nNodes) {
    el.textContent = "No calibration loaded.";
    return;
  }
  if (!report.worst) {
    el.textContent = `${report.nNodes} node(s) calibrated.`;
    return;
  }
  const worst = report.worst;
  const quality =
    worst.sigmas > 4 ? "well separated" : worst.sigmas > 2 ? "usable" : "poorly separated";
  let text =
    `${report.nNodes} nodes calibrated; closest pair ` +
    `(${worst.nodes[0]}, ${worst.nodes[1]}) is ${worst.sigmas.toFixed(1)} sigma apart ` +
    `- ${quality}.`;

  // A non-unimodal node invalidates the sigma figures above, so say so rather
  // than letting a reassuring number stand.
  if (report.nonUnimodal && report.nonUnimodal.length) {
    const nodes = report.nonUnimodal.map((d) => d.node).join(", ");
    text +=
      ` Warning: node(s) ${nodes} produced readings with more than one cluster,` +
      ` so these separation figures understate the real confusion. Recalibrate` +
      ` those nodes, touching each the same way throughout.`;
  }
  el.textContent = text;
}

/**
 * Warn when the audio output cannot reproduce a cue the spec depends on.
 */
function renderOutputWarnings() {
  const el = $("#outputWarning");
  if (!el) return;
  const warnings = engine.outputWarnings();
  el.textContent = warnings.join(" ");
}

// A clickable node list doubles as a keyboard-accessible way to explore the
// sonification without the physical object, which matters both for design
// iteration and for anyone who cannot handle the artifact.
function renderNodeButtons() {
  const el = $("#nodeButtons");
  if (!el || !spec) return;
  el.innerHTML = "";
  for (const node of spec.nodes) {
    const button = document.createElement("button");
    button.textContent = String(node);
    button.className = "node-button";
    button.setAttribute("aria-label", engine.describeNode(node));
    button.addEventListener("click", async () => {
      await engine.start();
      engine.playNode(node);
      showNode(node, 1.0, "manual");
      if (speakEnabled) engine.speakNode(node);
    });
    el.appendChild(button);
  }
}

function showNode(node, confidence, source) {
  setText("#selectedNode", node === NO_NODE ? "-" : `Node ${node}`);
  const meter = $("#confidenceBar");
  if (meter) meter.style.width = `${Math.round((confidence || 0) * 100)}%`;
  setText(
    "#confidenceText",
    node === NO_NODE ? "no touch" : `${Math.round(confidence * 100)}% confident (${source})`
  );
  const desc = $("#nodeDescription");
  if (desc) desc.textContent = node === NO_NODE ? "" : engine.describeNode(node);
  if (node !== NO_NODE) announce(engine.describeNode(node));

  // Highlight the corresponding button.
  for (const button of document.querySelectorAll(".node-button")) {
    button.classList.toggle("active", button.textContent === String(node));
  }
}

function handleReading(rawValue) {
  const result = classifier.push(rawValue);
  const { node, changed } = tracker.push(result.node);

  setText("#rawValue", Number.isFinite(result.value) ? result.value.toFixed(0) : "-");
  if (result.reason) setText("#rejectReason", result.reason);
  else setText("#rejectReason", "");

  if (changed) {
    lastNode = node;
    engine.setTouchedNode(node === NO_NODE ? null : node);
    showNode(node, result.confidence, "sensor");
    if (node !== NO_NODE && speakEnabled) engine.speakNode(node);
  }
}

/* ---------------------------------------------------------------- controls */

$("#startAudioButton")?.addEventListener("click", async () => {
  const state = await engine.start();
  setText("#audioState", `audio ${state}`);
  // Only knowable once an AudioContext exists.
  renderOutputWarnings();
  if (socket && !running) {
    socket.emit("io", { statuses: "1" });
    running = true;
  }
});

$("#modeSelect")?.addEventListener("change", (e) => {
  engine.mode = e.target.value;
  engine.stop();
});

$("#volumeSlider")?.addEventListener("input", (e) => {
  engine.setMasterGain(Number(e.target.value));
});

$("#speakToggle")?.addEventListener("change", (e) => {
  speakEnabled = e.target.checked;
});

$("#tourButton")?.addEventListener("click", async () => {
  await engine.start();
  await engine.playTour(null, {
    onNode: (id) => showNode(parseInt(id, 10), 1.0, "tour"),
  });
});

$("#neighbourButton")?.addEventListener("click", async () => {
  if (lastNode === NO_NODE) return;
  await engine.start();
  await engine.playNeighbourhood(lastNode, {
    onNode: (id) => showNode(parseInt(id, 10), 1.0, "neighbourhood"),
  });
});

$("#stopButton")?.addEventListener("click", () => {
  engine.stop();
  tracker.reset();
  showNode(NO_NODE, 0, "stopped");
});

if (socket) {
  socket.on("data", (data) => handleReading(data.capValue));
}

// Simulate a sensor stream so the sonification can be exercised with no
// hardware attached.
let simulationTimer = null;
$("#simulateButton")?.addEventListener("click", async () => {
  await engine.start();
  if (simulationTimer) {
    clearInterval(simulationTimer);
    simulationTimer = null;
    setText("#simulateButton", "Simulate touches");
    return;
  }
  setText("#simulateButton", "Stop simulation");
  const nodes = spec ? spec.nodes : [];
  let index = 0;
  simulationTimer = setInterval(() => {
    const node = nodes[index % nodes.length];
    index += 1;
    const profile = classifier.profiles.get(String(node));
    if (profile) {
      // Feed the real signal path so the classifier and tracker are exercised.
      for (let i = 0; i < 6; i++) {
        handleReading(profile.mean + (Math.random() - 0.5) * profile.sd * 2);
      }
    } else {
      engine.setTouchedNode(null);
      engine.setTouchedNode(node);
      showNode(node, 1.0, "simulated");
    }
  }, 900);
});

/* ------------------------------------------------------------------- boot  */

(async () => {
  try {
    await loadSpec();
    setText("#specState", `loaded ${spec.nodes.length} nodes`);
  } catch (e) {
    setText("#specState", "no sonification.json found - generate one from Python");
  }
  try {
    await loadConfig();
  } catch (e) {
    setText("#separation", "no config.json found - run calibration first");
  }
})();

export { engine, classifier, loadSpec, loadConfig };
