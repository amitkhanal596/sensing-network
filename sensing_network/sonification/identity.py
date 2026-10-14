"""
Making nodes identifiable, not just measurable.

A data-driven sonification has a blind spot that a visual encoding does not.
Looking at a network, position tells you *which* node you are looking at even
when two nodes share a value. Touching a physicalization, your hand knows
where it is -- but the sound does not: on a regular graph, every node has the
same degree, so every node plays the same tone, and the audio cannot confirm
what the hand found.

That matters most exactly where this project aims. A sighted user glances at
the object to resolve the ambiguity; a blind user cannot, so the audio has to
carry identity itself.

Two ways to add it, both of which leave the data mapping intact:

  1. `add_identity_channel` puts identity on a channel the data is not using.
     Cheap, but it spends a channel, and each channel supports only a few
     levels a listener can *identify* in isolation (see
     CHANNEL_IDENTIFY_CAPACITY), so it does not scale to many nodes.
  2. `add_identity_prefix` plays a short marker before the data tone. It costs
     time rather than a channel, and keeps the data tone undisturbed --
     the same idea as an earcon preceding a spoken value.

Which to prefer depends on the task: for comparing values across nodes, keep
the data tone clean and use a prefix; for confirming location during free
exploration, a channel is faster.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import numpy as np

from sensing_network.sonification.encoding import Channel, CHANNEL_SPECS
from sensing_network.sonification.scales import (
    LinearScale,
    OrdinalScale,
    QuantizedScale,
    midi_to_hz,
)


# How many levels of a channel a listener can reliably *identify* in
# isolation (not merely discriminate side by side). These are deliberately
# conservative: absolute identification along a single dimension runs out at
# roughly 5-7 categories (Miller 1956, citing Pollack on pitch), and stereo
# position is coarser still. Exceeding these produces a sonification that a
# metric will call distinct and a listener will not.
CHANNEL_IDENTIFY_CAPACITY = {
    "timbre": 4,  # the four oscillator waveforms available
    "pan": 5,
    "detune": 3,
    "attack": 3,
}


def _channel_capacity(name):
    return CHANNEL_IDENTIFY_CAPACITY.get(name, 3)


# Perceptual resolution per channel: the smallest difference that a listener
# can be expected to notice. Used to decide whether two tones are *audibly*
# different rather than merely numerically different.
#
# These are engineering approximations, not measured thresholds for this
# hardware. They exist so the checker errs toward "these may sound the same"
# instead of declaring success on floating-point inequality. Validating them
# needs a listening study (see CRITICAL_REVIEW.md 2).
CHANNEL_RESOLUTION = {
    # ~1 semitone: comfortably above the discrimination threshold, and near
    # the practical floor for ranking two tones heard at different times.
    "pitch": {"kind": "ratio", "value": 2 ** (1 / 12)},
    "brightness": {"kind": "absolute", "value": 0.15},
    "loudness": {"kind": "absolute", "value": 0.15},
    "duration": {"kind": "absolute", "value": 0.08},
    "pulse_rate": {"kind": "absolute", "value": 1.5},
    "tremolo_depth": {"kind": "absolute", "value": 0.2},
    "detune": {"kind": "absolute", "value": 10.0},
    "pan": {"kind": "absolute", "value": 0.35},
    "attack": {"kind": "absolute", "value": 0.03},
}


def _audibly_different(channel, a, b):
    """
    Whether two values on one channel differ enough to be heard apart.
    """
    if isinstance(a, str) or isinstance(b, str):
        return a != b
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        a_seq = list(a) if isinstance(a, (list, tuple)) else [a]
        b_seq = list(b) if isinstance(b, (list, tuple)) else [b]
        if len(a_seq) != len(b_seq):
            return True
        # A marker sequence differs if any of its notes differs audibly.
        return any(
            _audibly_different("pitch", x, y) for x, y in zip(a_seq, b_seq)
        )

    resolution = CHANNEL_RESOLUTION.get(channel)
    if resolution is None:
        return a != b
    if resolution["kind"] == "ratio":
        lo, hi = sorted((abs(float(a)), abs(float(b))))
        if lo <= 0:
            return hi > 0
        return (hi / lo) >= resolution["value"]
    return abs(float(a) - float(b)) >= resolution["value"]


def perceptual_distinctness(spec):
    """
    How many nodes are *audibly* distinguishable from every other node.

    Two tones count as different only when at least one channel differs by
    more than that channel's perceptual resolution -- not merely when their
    floating-point parameters differ. A design can therefore be numerically
    unique and still fail this check, which is the common case: linearly
    spreading 20 nodes across the stereo field produces 20 distinct numbers
    and far fewer distinguishable positions.

    This is a design aid, not evidence. Whether listeners actually separate
    these tones is an empirical question that needs a study.
    """
    tones = list(spec.tones)
    n = len(tones)
    channels = sorted({name for tone in tones for name in tone.params})

    collisions = []
    confusable_with = {tone.node_id: set() for tone in tones}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tones[i], tones[j]
            distinct = False
            for channel in channels:
                if channel in ("identity_prefix_duration", "identity_prefix_gap"):
                    continue
                if channel not in a.params or channel not in b.params:
                    distinct = True
                    break
                if _audibly_different(channel, a.params[channel], b.params[channel]):
                    distinct = True
                    break
            if not distinct:
                collisions.append([a.node_id, b.node_id])
                confusable_with[a.node_id].add(b.node_id)
                confusable_with[b.node_id].add(a.node_id)

    n_unique = sum(1 for node, others in confusable_with.items() if not others)
    worst = max((len(v) + 1 for v in confusable_with.values()), default=0) if collisions else 0

    return {
        "n_nodes": n,
        "n_uniquely_identifiable": n_unique,
        "fraction_identifiable": n_unique / max(n, 1),
        "confusable_pairs": collisions,
        "worst_group": worst,
    }


def identity_ambiguity(spec):
    """
    Report how many nodes have numerically distinct sound parameters.

    NOTE: this counts *parameter* distinctness, not perceptual distinctness.
    Two tones a hundredth of a hertz apart count as different here, and a
    listener will not hear them as different. Use `perceptual_distinctness`
    for a claim about what a person can tell apart; this function is retained
    for checking that the encoding itself assigns different values.

    Returns a dict with the number of distinct sounds, the fraction of nodes
    that are numerically distinct, and the groups that collide.
    """
    def hashable(value):
        # Round continuous parameters before comparing: two tones a hundredth
        # of a hertz apart are not distinguishable by ear. Some parameters
        # (an identity prefix, say) are sequences, so flatten those to tuples.
        if isinstance(value, (list, tuple)):
            return tuple(hashable(v) for v in value)
        if isinstance(value, bool) or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        return value

    signatures = {}
    for tone in spec.tones:
        key = tuple(
            sorted(
                (name, hashable(value)) for name, value in tone.params.items()
            )
        )
        signatures.setdefault(key, []).append(tone.node_id)

    collisions = [nodes for nodes in signatures.values() if len(nodes) > 1]
    n_unique = sum(1 for nodes in signatures.values() if len(nodes) == 1)

    return {
        "n_nodes": len(spec.tones),
        "n_distinct_sounds": len(signatures),
        "n_uniquely_identifiable": n_unique,
        "fraction_identifiable": n_unique / max(len(spec.tones), 1),
        "collisions": collisions,
        "worst_collision": max((len(c) for c in collisions), default=0),
    }


def _free_channels(spec):
    used = {ch.name for ch in spec.encoding.channels}
    # Ordered channels are reserved for data; identity is categorical, so
    # prefer channels that read as "which one" rather than "how much".
    #
    # Note on ordering: stereo pan is a strong identity cue but is inaudible
    # on mono output -- a single earbud, a laptop speaker, a phone -- which is
    # a common setup for a listener keeping one ear on the room. It is
    # therefore NOT first here. `ensure_identifiable` defaults to a prefix,
    # which survives any output; a caller who knows headphones are in use can
    # ask for pan explicitly.
    preference = ["timbre", "detune", "attack", "pan"]
    return [name for name in preference if name not in used]


def add_identity_channel(spec, channel=None, inplace=True):
    """
    Give each node a distinguishable value on an unused channel.

    Pan is available but is deliberately NOT the automatic default: stereo
    position is a strong identity cue and maps naturally onto a layout a hand
    is exploring, but it is inaudible on mono output -- a single earbud, a
    laptop or phone speaker -- which is a common setup for a listener keeping
    one ear on the room. Choose it explicitly when headphones are assured.
    The audio engine detects mono output and substitutes a pitched marker.

    Returns the modified spec.
    """
    target = spec if inplace else _copy_spec(spec)

    if channel is None:
        available = _free_channels(target)
        if not available:
            raise ValueError(
                "no free channel for identity; free one by reducing max_channels "
                "or use add_identity_prefix instead"
            )
        channel = available[0]

    nodes = list(target.nodes)
    n = len(nodes)

    if channel == "pan":
        lo, hi = CHANNEL_SPECS["pan"]["default_range"]
        values = np.linspace(lo, hi, n) if n > 1 else np.array([0.0])
    elif channel == "timbre":
        waveforms = list(CHANNEL_SPECS["timbre"]["default_range"])
        values = [waveforms[i % len(waveforms)] for i in range(n)]
    elif channel == "detune":
        lo, hi = CHANNEL_SPECS["detune"]["default_range"]
        values = np.linspace(lo, hi, n) if n > 1 else np.array([lo])
    else:
        lo, hi = CHANNEL_SPECS[channel]["default_range"]
        values = np.linspace(lo, hi, n) if n > 1 else np.array([lo])

    for tone, value in zip(target.tones, values):
        tone.params[channel] = value if isinstance(value, str) else float(value)

    # Record the identity channel in the encoding so the legend explains it
    # and the browser does not mistake it for a data mapping.
    identity_scale = OrdinalScale(
        domain=[int(x) if isinstance(x, (int, np.integer)) else x for x in nodes],
        range_=[v if isinstance(v, str) else float(v) for v in values],
    )
    identity_channel = Channel(
        channel,
        "node_id",
        identity_scale,
        label="node identity",
        meaning="which node is being touched",
    )
    target.encoding.add(identity_channel)
    target.metadata["identity_channel"] = channel
    return target


def add_identity_prefix(
    spec,
    pitch_set="pentatonic",
    root_midi=72,
    n_octaves=2,
    duration=0.09,
    gap=0.04,
    inplace=True,
):
    """
    Prepend a short identifying marker to each node's sound.

    The marker is a brief, quiet tone drawn from a register above the data
    tones, so it reads as punctuation rather than as data. Nodes are spread
    across the available pitches; with more nodes than pitches, a two-note
    marker is used, which covers n^2 nodes.

    The data tone is left exactly as it was.
    """
    target = spec if inplace else _copy_spec(spec)

    n = len(target.nodes)
    scale = QuantizedScale(
        pitch_set=pitch_set, root_midi=root_midi, n_octaves=n_octaves
    )
    available = [float(f) for f in midi_to_hz(scale.pitches_midi())]

    markers = []
    if n <= len(available):
        for i in range(n):
            markers.append([available[i]])
    else:
        # Two-note markers, ordered so adjacent nodes differ in the first note
        # (the one a listener hears soonest).
        for i in range(n):
            first = available[i % len(available)]
            second = available[(i // len(available)) % len(available)]
            markers.append([first, second])

    for tone, marker in zip(target.tones, markers):
        tone.params["identity_prefix"] = marker
        tone.params["identity_prefix_duration"] = duration
        tone.params["identity_prefix_gap"] = gap

    target.metadata["identity_prefix"] = {
        "pitch_set": pitch_set,
        "root_midi": root_midi,
        "duration": duration,
        "gap": gap,
        "n_notes": 1 if n <= len(available) else 2,
    }
    return target


def ensure_identifiable(spec, strategy="auto", inplace=True):
    """
    Add identity information only where the data mapping fails to provide it.

    `strategy` is "channel", "prefix", or "auto". Under "auto" a channel cue is
    used only when the channel can carry the node count with levels a listener
    can actually identify; otherwise a prefix is used, which works on any
    output device including mono.

    Ambiguity is judged perceptually (see `perceptual_distinctness`), not by
    floating-point inequality, so a design that merely assigns different
    numbers still counts as ambiguous.

    Returns (spec, report) where report describes what was ambiguous and what
    was done about it.
    """
    before = perceptual_distinctness(spec)

    if before["fraction_identifiable"] >= 1.0:
        return spec, {
            "before": before,
            "after": before,
            "action": "none",
            "reason": "every node already sounds distinct",
        }
    n_ambiguous = before["n_nodes"] - before["n_uniquely_identifiable"]

    if strategy == "auto":
        # A prefix is the safe default: it costs a little time but works on
        # every output device, whereas a channel cue can be silent (pan on
        # mono) or perceptually crowded (many nodes on one dimension).
        # A channel is used only when it can carry the node count with cues
        # a listener can actually tell apart.
        strategy = "prefix"
        free = _free_channels(spec)
        if free and _channel_capacity(free[0]) >= len(spec.nodes):
            strategy = "channel"

    if strategy == "channel":
        target = add_identity_channel(spec, inplace=inplace)
    elif strategy == "prefix":
        target = add_identity_prefix(spec, inplace=inplace)
    else:
        raise ValueError(f"unknown strategy '{strategy}'")

    after = perceptual_distinctness(target)
    return target, {
        "before": before,
        "after": after,
        "action": strategy,
        "reason": (
            f"{n_ambiguous} of {before['n_nodes']} nodes were not audibly "
            f"distinguishable from another node"
        ),
        # Say plainly when the cue did not fully solve the problem, rather
        # than reporting the action as if it had.
        "resolved": after["fraction_identifiable"] >= 1.0,
    }


def _copy_spec(spec):
    from sensing_network.sonification.encoding import SonificationSpec

    return SonificationSpec.from_dict(spec.to_dict())
