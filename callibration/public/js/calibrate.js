/*
Optional enhanced calibration, kept separate from the original page.

The published calibration flow (index.html + js/main.js) is left exactly as it
shipped with the paper. This file is an alternative that additionally records
each node's spread and flags data that breaks the classifier's assumptions,
writing a richer config that the explorer can use. Both formats are readable by
TouchClassifier, so the two flows interoperate.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
*/

import { TouchClassifier, StableNodeTracker, NO_NODE } from "./classifier.js";

const socket = typeof io !== "undefined" ? io() : null;

const classifier = new TouchClassifier({ driftAlpha: 0.02 });
const tracker = new StableNodeTracker();

const collected = [];
let mode = null;
let currentNode = -1;
let startTime = null;
let intervalId = null;

const $ = (sel) => document.querySelector(sel);
const setText = (sel, text) => {
  const el = $(sel);
  if (el) el.textContent = text;
};

function summarise() {
  classifier.fit(collected);
  const config = classifier.toConfig();
  const report = classifier.separationReport();

  const lines = [];
  lines.push(`${report.nNodes} nodes calibrated.`);

  if (report.worst) {
    const sigmas = report.worst.sigmas;
    const verdict =
      sigmas > 4 ? "well separated" : sigmas > 2 ? "usable" : "poorly separated";
    lines.push(
      `Closest pair (${report.worst.nodes[0]}, ${report.worst.nodes[1]}): ` +
        `${sigmas.toFixed(1)} sigma - ${verdict}.`
    );
    if (sigmas <= 2) {
      lines.push(
        "Those two nodes will be confused in use. Re-run calibration, or " +
          "regenerate the network with a wider resistance range."
      );
    }
  }

  // A node whose readings form more than one cluster breaks the single-Gaussian
  // model, and the sigma figures above will look better than reality.
  if (report.nonUnimodal && report.nonUnimodal.length) {
    const names = report.nonUnimodal.map((d) => d.node).join(", ");
    lines.push(
      `Warning: node(s) ${names} produced readings in more than one cluster, ` +
        `so the separation figures above understate the real confusion. ` +
        `Recalibrate them, touching each node the same way throughout.`
    );
  }

  setText("#status", lines.join(" "));

  if (socket) socket.emit("endProcessing", JSON.stringify(config));
  return config;
}

$("#startButton")?.addEventListener("click", () => {
  if (!socket) {
    setText("#status", "No sensor connection: run this page from the calibration server.");
    return;
  }
  mode = "collection";
  collected.length = 0;
  currentNode = -1;
  startTime = performance.now();
  socket.emit("io", { statuses: "1" });

  const nNodes = parseInt($("#nNodesInput").value, 10);
  const dwell = parseInt($("#dwellInput").value, 10) || 5000;

  setText("#status", "Get ready - do not touch the network yet.");

  intervalId = setInterval(() => {
    currentNode += 1;
    if (currentNode < nNodes) {
      setText("#status", `Touch node ${currentNode} of ${nNodes - 1}.`);
    } else {
      clearInterval(intervalId);
      if (socket) socket.emit("endCollection", JSON.stringify(collected));
      summarise();
      mode = null;
    }
  }, dwell);
});

$("#testButton")?.addEventListener("click", () => {
  if (!socket) return;
  mode = "test";
  tracker.reset();
  socket.emit("io", { statuses: "1" });
  setText("#status", "Touch any node to test the calibration.");
});

if (socket) {
  socket.on("data", (data) => {
    const value = data.capValue;
    if (mode === "collection") {
      collected.push({
        node: currentNode,
        time: performance.now() - startTime,
        value,
      });
      return;
    }
    if (mode === "test") {
      const result = classifier.push(value);
      const { node, changed } = tracker.push(result.node);
      if (changed) {
        setText(
          "#detected",
          node === NO_NODE
            ? "no touch"
            : `Node ${node} (${Math.round(result.confidence * 100)}% confident)`
        );
      }
    }
  });
}

export { classifier, summarise };
