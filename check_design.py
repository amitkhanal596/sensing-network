"""
Check a sonification design before committing it to a print.

A print takes hours and a calibration pass takes minutes, so it is worth
knowing in advance whether a design is actually listenable. This reports the
things that are cheap to check computationally and expensive to discover after
the fact:

  * nodes that sound identical, so a listener cannot tell them apart;
  * pitch steps too small to hear as different;
  * channels carrying almost no information because the attribute is nearly
    constant;
  * more simultaneous channels than a listener can attend to.

Usage:
    python check_design.py result/sonification.json

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import argparse
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sensing_network.sonification.encoding import SonificationSpec
from sensing_network.sonification.identity import (
    identity_ambiguity,
    perceptual_distinctness,
)
from sensing_network.sonification.scales import hz_to_midi

# Two different psychophysical questions, often conflated:
#
#   discrimination - "are these two tones the same?", heard close together.
#                    Thresholds are tiny (a fraction of a semitone).
#   identification - "which value is this?", heard alone. This is what a
#                    person touching a single node actually does, and it is
#                    far harder: absolute identification along one dimension
#                    saturates at roughly 5-7 categories (Miller 1956,
#                    citing Pollack's pitch data).
#
# A checker that tests discrimination will pass designs nobody can read, so
# both are reported, and the identification limit drives the verdict.
MIN_DISCRIMINABLE_SEMITONES = 0.5
COMFORTABLE_SEMITONES = 1.0

# Miller's "seven plus or minus two", with the conservative end taken because
# listeners here are also attending to touch. Not a hardware-validated number.
MAX_IDENTIFIABLE_LEVELS = 7
COMFORTABLE_IDENTIFIABLE_LEVELS = 5

# Attending to many simultaneous data streams is unreliable, and how many
# depends strongly on the cues separating them, so this is a soft guide rather
# than a measured limit.
MAX_COMFORTABLE_CHANNELS = 4


def check(spec):
    findings = []

    # 1. Can every node be told apart -- audibly, not just numerically?
    perceptual = perceptual_distinctness(spec)
    numeric = identity_ambiguity(spec)

    if perceptual["fraction_identifiable"] < 1.0:
        n_ambiguous = (
            perceptual["n_nodes"] - perceptual["n_uniquely_identifiable"]
        )
        examples = ", ".join(
            f"{a}/{b}" for a, b in perceptual["confusable_pairs"][:4]
        )
        detail = f" e.g. {examples}" if examples else ""
        hint = (
            "Enable ensure_identity."
            if numeric["fraction_identifiable"] < 1.0
            else "The parameters differ numerically but by less than a "
            "listener can hear; use identity_strategy='prefix' or fewer nodes "
            "per channel."
        )
        findings.append(
            (
                "error",
                f"{n_ambiguous} of {perceptual['n_nodes']} nodes are not "
                f"audibly distinguishable from another node{detail}. {hint}",
            )
        )
    else:
        findings.append(
            (
                "ok",
                f"all {perceptual['n_nodes']} nodes differ audibly on at "
                f"least one channel",
            )
        )

    # 2. Are the pitch steps large enough to hear?
    pitches = sorted(
        {
            round(float(t.params["pitch"]), 4)
            for t in spec.tones
            if "pitch" in t.params
        }
    )
    if len(pitches) > 1:
        midi = hz_to_midi(np.array(pitches))
        gaps = np.diff(midi)
        smallest = float(gaps.min())

        # (a) spacing: can two pitches be told apart at all?
        if smallest < MIN_DISCRIMINABLE_SEMITONES:
            findings.append(
                (
                    "error",
                    f"the closest two pitches are {smallest:.2f} semitones apart, "
                    f"below the ~{MIN_DISCRIMINABLE_SEMITONES} semitone floor for "
                    f"telling two tones apart. Reduce n_pitch_steps or widen "
                    f"n_octaves.",
                )
            )
        elif smallest < COMFORTABLE_SEMITONES:
            findings.append(
                (
                    "warn",
                    f"the closest two pitches are {smallest:.2f} semitones apart; "
                    f"they will be hard to rank without hearing them together.",
                )
            )
        else:
            findings.append(
                (
                    "ok",
                    f"pitch spacing is {smallest:.1f} semitones at the closest pair",
                )
            )

        # (b) count: can a value be *named* from a tone heard alone? This is
        # the task a person touching one node performs, and it is the limit
        # that actually binds.
        if len(pitches) > MAX_IDENTIFIABLE_LEVELS:
            findings.append(
                (
                    "error",
                    f"{len(pitches)} distinct pitch levels are used, but "
                    f"absolute identification of a tone heard alone saturates "
                    f"around {MAX_IDENTIFIABLE_LEVELS} levels. A listener will "
                    f"hear 'higher' and 'lower' but cannot recover the value. "
                    f"Set n_pitch_steps to about "
                    f"{COMFORTABLE_IDENTIFIABLE_LEVELS}.",
                )
            )
        elif len(pitches) > COMFORTABLE_IDENTIFIABLE_LEVELS:
            findings.append(
                (
                    "warn",
                    f"{len(pitches)} pitch levels is near the limit of what can "
                    f"be identified in isolation (~"
                    f"{COMFORTABLE_IDENTIFIABLE_LEVELS}); ordering will read "
                    f"more reliably than exact values.",
                )
            )
        else:
            findings.append(
                ("ok", f"{len(pitches)} pitch levels, within the identifiable range")
            )
    elif len(pitches) == 1:
        findings.append(
            (
                "warn",
                "every node plays the same pitch; the primary attribute is "
                "constant or nearly so, so pitch is carrying no information.",
            )
        )

    # 3. Is any channel wasted on a near-constant attribute?
    for channel in spec.encoding.channels:
        if channel.attribute == "node_id":
            continue
        values = spec.attribute_data.get(channel.attribute)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if len(np.unique(np.round(values, 9))) == 1:
            findings.append(
                (
                    "warn",
                    f"'{channel.attribute}' is constant, so the "
                    f"{channel.name} channel conveys nothing. Map a different "
                    f"attribute or drop the channel.",
                )
            )

    # 3b. Was an automatic scale transform applied? It changes the reading and
    # is otherwise invisible.
    transform = spec.metadata.get("pitch_scale_transform")
    if transform == "log":
        findings.append(
            (
                "warn",
                "the primary attribute was automatically put on a log scale "
                "because it is heavily skewed; equal pitch steps therefore "
                "mean equal *ratios*, not equal differences. State this in any "
                "caption.",
            )
        )

    # 4. How many channels compete for attention?
    data_channels = [c for c in spec.encoding.channels if c.attribute != "node_id"]
    distinct_attributes = {c.attribute for c in data_channels}
    if len(distinct_attributes) > MAX_COMFORTABLE_CHANNELS:
        findings.append(
            (
                "warn",
                f"{len(distinct_attributes)} attributes are encoded at once; "
                f"beyond about {MAX_COMFORTABLE_CHANNELS} a listener cannot "
                f"attend to them all.",
            )
        )
    else:
        findings.append(
            (
                "ok",
                f"{len(distinct_attributes)} attribute(s) encoded across "
                f"{len(data_channels)} channel(s)",
            )
        )

    # 5. Does a redundant pairing actually agree?
    by_attribute = {}
    for channel in data_channels:
        by_attribute.setdefault(channel.attribute, []).append(channel.name)
    for attribute, names in by_attribute.items():
        if len(names) > 1:
            findings.append(
                ("ok", f"'{attribute}' is encoded redundantly ({', '.join(names)})")
            )

    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="path to a sonification.json")
    args = parser.parse_args()

    spec = SonificationSpec.load(args.spec)
    findings = check(spec)

    symbols = {"ok": "  ok  ", " warn": "warn", "error": "ERROR "}
    print(f"\nchecking {args.spec}\n")
    for level, message in findings:
        label = {"ok": "  ok  ", "warn": " warn ", "error": "ERROR "}[level]
        print(f"[{label}] {message}")

    n_errors = sum(1 for level, _ in findings if level == "error")
    n_warnings = sum(1 for level, _ in findings if level == "warn")
    print(f"\n{n_errors} error(s), {n_warnings} warning(s)")
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
