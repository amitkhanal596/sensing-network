"""
Tests for the sonification layer.

Run with:  python -m pytest tests/ -q
       or: python tests/test_sonification.py

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sensing_network.sonification import (
    default_sonification,
    node_attributes,
    LinearScale,
    LogScale,
    OrdinalScale,
    QuantizedScale,
    midi_to_hz,
    hz_to_midi,
)
from sensing_network.sonification.encoding import SonificationSpec, CHANNEL_SPECS
from sensing_network.sonification.forms import (
    line_graph_form,
    bar_chart_form,
    scatter_form,
    region_adjacency_form,
    sonify_form,
)

TRIANGLE_PAIR = {
    "nodes": list(range(6)),
    "links": [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [3, 5], [4, 5]],
}


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


# ------------------------------------------------------------------ scales


def test_midi_hz_roundtrip():
    for midi in [21, 60, 69, 96]:
        _assert(
            abs(hz_to_midi(midi_to_hz(midi)) - midi) < 1e-9,
            f"midi/hz round trip failed at {midi}",
        )
    _assert(abs(midi_to_hz(69) - 440.0) < 1e-9, "A4 must be 440 Hz")


def test_linear_scale_clamps():
    scale = LinearScale(domain=(0, 10), range_=(100, 200))
    _assert(scale([-5])[0] == 100, "values below the domain must clamp")
    _assert(scale([15])[0] == 200, "values above the domain must clamp")
    _assert(abs(scale([5])[0] - 150) < 1e-9, "midpoint must map to mid-range")


def test_constant_attribute_does_not_divide_by_zero():
    scale = LinearScale(range_=(0, 1))
    out = scale([7.0, 7.0, 7.0])
    _assert(np.all(np.isfinite(out)), "a constant attribute must stay finite")


def test_log_scale_handles_non_positive():
    scale = LogScale(range_=(0, 1))
    out = scale([0.0, 1.0, 10.0, 100.0])
    _assert(np.all(np.isfinite(out)), "log scale must handle zero")
    _assert(out[0] < out[-1], "log scale must stay increasing")


def test_quantized_scale_stays_in_octave_range():
    scale = QuantizedScale(pitch_set="pentatonic", root_midi=60, n_octaves=2)
    notes = scale.pitches_midi()
    _assert(min(notes) == 60, "scale must start at the root")
    _assert(max(notes) == 84, "scale must not exceed n_octaves above the root")


def test_quantized_scale_respects_n_steps():
    scale = QuantizedScale(pitch_set="chromatic", n_octaves=2, n_steps=5)
    _assert(len(scale.pitches_midi()) == 5, "n_steps must cap the pitch count")


def test_ordinal_scale_cycles():
    scale = OrdinalScale(domain=["a", "b", "c"], range_=["x", "y"])
    _assert(scale(["a", "b", "c"]) == ["x", "y", "x"], "ordinal range must cycle")


# -------------------------------------------------------------- attributes


def test_node_attributes_align_to_nodes():
    data = node_attributes(
        TRIANGLE_PAIR["nodes"], TRIANGLE_PAIR["links"], attributes=["degree"]
    )
    _assert(len(data["degree"]) == 6, "attributes must align to the node list")
    _assert(data["degree"][2] == 3, "node 2 has three links in this graph")


def test_unknown_attribute_raises():
    try:
        node_attributes([0, 1], [[0, 1]], attributes=["not_a_real_metric"])
    except ValueError:
        return
    raise AssertionError("an unknown attribute must raise ValueError")


def test_extra_attribute_length_is_checked():
    try:
        node_attributes([0, 1, 2], [[0, 1], [1, 2]], attributes=[], extra={"v": [1, 2]})
    except ValueError:
        return
    raise AssertionError("a mismatched extra attribute must raise ValueError")


def test_disconnected_graph_does_not_crash():
    # Eigenvector centrality fails to converge here; it must degrade, not raise.
    spec = default_sonification(
        [0, 1, 2, 3], [[0, 1], [2, 3]], primary="eigenvector", categorical=False
    )
    _assert(len(spec.tones) == 4, "a disconnected graph must still produce tones")


# ---------------------------------------------------------------- encoding


def test_every_node_gets_a_tone():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    _assert(len(spec.tones) == 6, "each node needs exactly one tone")
    _assert(
        [t.node_id for t in spec.tones] == TRIANGLE_PAIR["nodes"],
        "tone order must match node order, since sensing returns node IDs",
    )


def test_pitch_is_monotonic_in_the_primary_attribute():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    pairs = sorted(
        (t.attributes["degree"], t.params["pitch"]) for t in spec.tones
    )
    pitches = [p for _, p in pairs]
    _assert(
        all(a <= b for a, b in zip(pitches, pitches[1:])),
        "higher degree must never sound lower in pitch",
    )


def test_channel_values_stay_in_declared_range():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree", secondary="betweenness")
    for channel in spec.encoding.channels:
        spec_range = CHANNEL_SPECS[channel.name]["default_range"]
        if not CHANNEL_SPECS[channel.name]["ordered"]:
            continue
        for tone in spec.tones:
            value = tone.params[channel.name]
            _assert(
                spec_range[0] - 1e-6 <= value <= spec_range[1] + 1e-6,
                f"{channel.name}={value} escaped its range {spec_range}",
            )


def test_timbre_is_categorical_and_valid():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    allowed = set(CHANNEL_SPECS["timbre"]["default_range"])
    for tone in spec.tones:
        if "timbre" in tone.params:
            _assert(
                tone.params["timbre"] in allowed,
                f"timbre {tone.params['timbre']} is not a valid waveform",
            )


def test_max_channels_is_respected():
    spec = default_sonification(
        **TRIANGLE_PAIR,
        primary="degree",
        secondary="betweenness",
        max_channels=2,
        ensure_identity=False,
    )
    _assert(len(spec.encoding.channels) <= 2, "max_channels must be honoured")


def test_max_channels_bounds_data_channels_only():
    # An identity cue is not a data mapping, so it may exceed the data budget;
    # what must hold is that no extra *data* channel sneaks in.
    spec = default_sonification(
        **TRIANGLE_PAIR, primary="degree", secondary="betweenness", max_channels=2
    )
    data_channels = [c for c in spec.encoding.channels if c.attribute != "node_id"]
    _assert(len(data_channels) <= 2, "max_channels must bound data channels")


def test_identity_makes_a_regular_graph_distinguishable():
    from sensing_network.sonification.identity import identity_ambiguity

    # In K4 every node has degree 3, so a degree mapping alone gives every
    # node the same sound.
    k4_nodes = [0, 1, 2, 3]
    k4_links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    plain = default_sonification(
        k4_nodes, k4_links, primary="degree", ensure_identity=False
    )
    _assert(
        identity_ambiguity(plain)["n_distinct_sounds"] == 1,
        "K4 with a degree mapping should collapse to one sound",
    )

    fixed = default_sonification(k4_nodes, k4_links, primary="degree")
    _assert(
        identity_ambiguity(fixed)["fraction_identifiable"] == 1.0,
        "identity must make every node distinguishable",
    )


def test_identity_prefix_strategy_is_distinguishable():
    from sensing_network.sonification.identity import ensure_identifiable, identity_ambiguity

    nodes = list(range(6))
    links = [[i, (i + 1) % 6] for i in range(6)]
    spec = default_sonification(nodes, links, primary="degree", ensure_identity=False)
    fixed, report = ensure_identifiable(spec, strategy="prefix")
    _assert(report["action"] == "prefix", "the requested strategy must be used")
    _assert(
        identity_ambiguity(fixed)["fraction_identifiable"] == 1.0,
        "a prefix must make every node distinguishable",
    )
    for tone in fixed.tones:
        _assert(
            isinstance(tone.params["identity_prefix"], list),
            "a prefix must be a list of frequencies the player can schedule",
        )


def test_identity_does_not_disturb_the_data_mapping():
    from sensing_network.sonification.identity import ensure_identifiable

    spec = default_sonification(**TRIANGLE_PAIR, primary="degree", ensure_identity=False)
    before = [t.params["pitch"] for t in spec.tones]
    fixed, _ = ensure_identifiable(spec, strategy="prefix")
    after = [t.params["pitch"] for t in fixed.tones]
    _assert(before == after, "adding identity must not change the data tones")


def test_json_roundtrip_preserves_tones():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree", secondary="betweenness")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spec.json")
        spec.save(path)
        with open(path) as f:
            raw = json.load(f)
        restored = SonificationSpec.from_dict(raw)
    _assert(
        [t.params for t in restored.tones] == [t.params for t in spec.tones],
        "a saved spec must reload identically -- the browser depends on it",
    )


def test_spec_json_is_serializable():
    # numpy types are not JSON-serializable; the spec must convert them.
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    json.dumps(spec.to_dict())


def test_legend_and_description_are_non_empty():
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    _assert(len(spec.legend()) == len(spec.encoding.channels), "one legend line per channel")
    text = spec.describe_node(2)
    _assert("Node 2" in text, "a node description must name its node")


def test_description_does_not_repeat_shared_attributes():
    # Pitch and brightness both encode the primary attribute by default.
    spec = default_sonification(**TRIANGLE_PAIR, primary="degree")
    text = spec.describe_node(2)
    _assert(text.count("degree") == 1, f"attribute repeated in description: {text}")


def test_missing_extra_attribute_raises():
    try:
        default_sonification(**TRIANGLE_PAIR, primary="revenue")
    except ValueError:
        return
    raise AssertionError("an unsupplied primary attribute must raise ValueError")


def test_log_used_for_skewed_attributes():
    # A single large outlier would, on a linear scale, push every other value
    # onto the bottom pitch. The log transform must spread them out. Compare
    # the two directly rather than asserting a pitch count, which also depends
    # on how many steps the scale is allowed to use.
    values = np.array([1, 1, 1, 2, 2, 900.0])
    kwargs = dict(
        primary="pop",
        extra_attributes={"pop": values},
        categorical=False,
        ensure_identity=False,
        n_pitch_steps=None,
    )
    spec = default_sonification(**TRIANGLE_PAIR, **kwargs)
    _assert(
        spec.metadata["pitch_scale_transform"] == "log",
        "this attribute is skewed enough to warrant a log scale",
    )

    log_pitches = {round(t.params["pitch"], 3) for t in spec.tones}
    # What a linear scale would have produced over the same domain.
    linear = QuantizedScale()
    linear.fit(values)
    linear_pitches = {round(p, 3) for p in linear(values)}
    _assert(
        len(log_pitches) > len(linear_pitches),
        f"the log scale should separate more values than linear "
        f"({len(log_pitches)} vs {len(linear_pitches)})",
    )


def test_pitch_steps_default_respects_identification_limit():
    from sensing_network.sonification.pipeline import DEFAULT_PITCH_STEPS

    nodes = list(range(30))
    links = [[i, i + 1] for i in range(29)]
    spec = default_sonification(
        nodes, links, primary="v",
        extra_attributes={"v": np.arange(30.0)},
        categorical=False, ensure_identity=False,
    )
    distinct = {round(t.params["pitch"], 3) for t in spec.tones}
    _assert(
        len(distinct) <= DEFAULT_PITCH_STEPS,
        f"default should cap pitch levels at {DEFAULT_PITCH_STEPS}, "
        f"got {len(distinct)}",
    )


def test_pitch_steps_can_be_uncapped_deliberately():
    nodes = list(range(30))
    links = [[i, i + 1] for i in range(29)]
    spec = default_sonification(
        nodes, links, primary="v",
        extra_attributes={"v": np.arange(30.0)},
        categorical=False, ensure_identity=False, n_pitch_steps=None,
    )
    distinct = {round(t.params["pitch"], 3) for t in spec.tones}
    _assert(len(distinct) > 5, "an explicit None must lift the cap")


# ------------------------------------------------------------------- forms


def test_line_graph_form_is_a_path():
    form = line_graph_form([3, 7, 2, 9])
    _assert(form["links"] == [[0, 1], [1, 2], [2, 3]], "a line graph must be a path")
    _assert(len(form["node_positions"]) == 4, "one position per point")


def test_line_graph_needs_two_points():
    try:
        line_graph_form([1])
    except ValueError:
        return
    raise AssertionError("a one-point line graph must raise ValueError")


def test_scatter_mst_is_a_spanning_tree():
    points = np.random.RandomState(0).rand(9, 2) * 100
    form = scatter_form(points)
    _assert(
        len(form["links"]) == len(points) - 1,
        "an MST over n points has exactly n-1 links",
    )


def test_scatter_knn_is_connected():
    import networkx as nx

    points = np.random.RandomState(1).rand(12, 2) * 100
    form = scatter_form(points, connect="knn", k=1)
    graph = nx.Graph()
    graph.add_nodes_from(form["nodes"])
    graph.add_edges_from([tuple(l) for l in form["links"]])
    _assert(
        nx.is_connected(graph),
        "the sensing chain requires one component, so k-NN must be stitched up",
    )


def test_region_adjacency_maps_names_to_ids():
    form = region_adjacency_form(
        {"a": ["b"], "b": ["c"]}, values={"a": 1, "b": 2, "c": 3}
    )
    _assert(form["region_names"] == ["a", "b", "c"], "region order must be stable")
    _assert(len(form["nodes"]) == 3, "one node per region")


def test_sonify_form_uses_the_data_attribute():
    form = line_graph_form([3, 7, 2, 9], label="temperature")
    spec = sonify_form(form)
    _assert(
        spec.encoding.channels[0].attribute == "temperature",
        "a line graph must sonify its values, not its graph degree",
    )
    lowest = min(spec.tones, key=lambda t: t.attributes["temperature"])
    highest = max(spec.tones, key=lambda t: t.attributes["temperature"])
    _assert(
        lowest.params["pitch"] < highest.params["pitch"],
        "the largest value must sound highest",
    )


def test_bar_chart_form_positions_match_values():
    form = bar_chart_form([10, 40, 25])
    heights = form["node_positions"][:, 1]
    _assert(heights[1] > heights[2] > heights[0], "bar heights must follow values")


# ------------------------------------------------------- polarity (Walker)


def test_polarity_inverts_the_pitch_mapping():
    nodes = list(range(6))
    links = [[i, i + 1] for i in range(5)]
    values = np.arange(6.0)

    rising = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": values},
        categorical=False, polarity=1, ensure_identity=False,
    )
    falling = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": values},
        categorical=False, polarity=-1, ensure_identity=False,
    )
    rising_pitches = [t.params["pitch"] for t in rising.tones]
    falling_pitches = [t.params["pitch"] for t in falling.tones]
    _assert(rising_pitches[0] < rising_pitches[-1], "polarity +1 must rise")
    _assert(falling_pitches[0] > falling_pitches[-1], "polarity -1 must fall")


def test_size_attributes_default_to_negative_polarity():
    # Walker (2002): listeners map increasing pitch to *decreasing* size.
    nodes = list(range(5))
    links = [[i, i + 1] for i in range(4)]
    spec = default_sonification(
        nodes, links, primary="node_size",
        extra_attributes={"node_size": np.arange(5.0)},
        categorical=False, ensure_identity=False,
    )
    _assert(spec.metadata["polarity"] == -1, "size should default to negative polarity")
    pitches = [t.params["pitch"] for t in spec.tones]
    _assert(pitches[0] > pitches[-1], "the largest size should sound lowest")


def test_legend_states_the_actual_polarity():
    nodes = list(range(5))
    links = [[i, i + 1] for i in range(4)]
    spec = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": np.arange(5.0)},
        categorical=False, polarity=-1, ensure_identity=False,
    )
    legend = " ".join(spec.legend())
    _assert(
        "lower pitch" in legend,
        f"an inverted mapping must say so in the legend: {legend}",
    )


def test_brightness_follows_the_same_polarity_as_pitch():
    nodes = list(range(5))
    links = [[i, i + 1] for i in range(4)]
    spec = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": np.arange(5.0)},
        categorical=False, polarity=-1, redundant_brightness=True,
        ensure_identity=False,
    )
    pitches = [t.params["pitch"] for t in spec.tones]
    brightness = [t.params["brightness"] for t in spec.tones]
    _assert(
        (pitches[0] > pitches[-1]) == (brightness[0] > brightness[-1]),
        "a redundant channel that contradicts pitch would be worse than none",
    )


def test_polarity_survives_a_json_roundtrip():
    nodes = list(range(5))
    links = [[i, i + 1] for i in range(4)]
    spec = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": np.arange(5.0)},
        categorical=False, polarity=-1, ensure_identity=False,
    )
    restored = SonificationSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    _assert(
        [t.params["pitch"] for t in restored.tones]
        == [t.params["pitch"] for t in spec.tones],
        "polarity must survive serialization",
    )


# ------------------------------------------- perceptual distinctness (2)


def test_perceptual_distinctness_is_stricter_than_numeric():
    from sensing_network.sonification.identity import (
        perceptual_distinctness,
        identity_ambiguity,
        add_identity_channel,
    )

    # 20 nodes spread linearly across the stereo field: 20 distinct numbers,
    # far fewer distinguishable positions.
    nodes = list(range(20))
    links = [[i, i + 1] for i in range(19)]
    base = default_sonification(nodes, links, primary="degree", ensure_identity=False)
    panned = add_identity_channel(base, channel="pan", inplace=False)

    numeric = identity_ambiguity(panned)["fraction_identifiable"]
    perceptual = perceptual_distinctness(panned)["fraction_identifiable"]
    _assert(numeric == 1.0, "all pan values are numerically distinct")
    _assert(
        perceptual < numeric,
        "the perceptual metric must not simply repeat the numeric one",
    )


def test_perceptual_metric_accepts_clearly_distinct_tones():
    from sensing_network.sonification.identity import perceptual_distinctness

    nodes = list(range(4))
    links = [[i, i + 1] for i in range(3)]
    spec = default_sonification(
        nodes, links, primary="v", extra_attributes={"v": np.arange(4.0)},
        categorical=False, ensure_identity=False,
    )
    result = perceptual_distinctness(spec)
    _assert(
        result["fraction_identifiable"] == 1.0,
        "octave-scale pitch differences must count as distinguishable",
    )


def test_default_identity_is_perceptually_sound_at_scale():
    from sensing_network.sonification.identity import perceptual_distinctness

    nodes = list(range(20))
    links = [[i, i + 1] for i in range(19)]
    spec = default_sonification(nodes, links, primary="degree")
    _assert(
        perceptual_distinctness(spec)["fraction_identifiable"] == 1.0,
        "the default identity strategy must work for a realistic node count",
    )


def test_identity_strategy_scales_with_channel_capacity():
    # A channel cue can only carry a few identifiable levels, so a large
    # network must fall back to a prefix rather than overfilling a channel.
    small = default_sonification(
        [0, 1, 2, 3], [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        primary="degree",
    )
    large = default_sonification(
        list(range(20)), [[i, i + 1] for i in range(19)], primary="degree"
    )
    _assert(
        large.metadata["identity"]["action"] == "prefix",
        "20 nodes exceed any single channel's identification capacity",
    )
    _assert(
        small.metadata["identity"]["action"] in ("channel", "prefix"),
        "a small network may use either strategy",
    )


def test_log_transform_is_recorded_in_metadata():
    # The log switch is automatic and changes the reading, so it must be
    # visible to anyone auditing a generated figure.
    nodes = list(range(6))
    links = [[i, i + 1] for i in range(5)]
    skewed = np.array([1.0, 1.0, 2.0, 3.0, 50.0, 900.0])
    spec = default_sonification(
        nodes, links, primary="pop", extra_attributes={"pop": skewed},
        categorical=False, ensure_identity=False,
    )
    _assert(
        spec.metadata["pitch_scale_transform"] == "log",
        "a skewed attribute should be recorded as log-scaled",
    )
    _assert(
        "log_thresholds" in spec.metadata,
        "the thresholds that triggered the switch must be recorded",
    )


def run_all():
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_all())
