"""
Automatic construction of a sonification from a network.

This pipeline should not require its user to know psychoacoustics, so the
default picks an encoding on their behalf. The rules below are *design
defaults informed by the sonification literature*, not validated findings for
this hardware; where the literature is contested that is said explicitly,
because these comments are the closest thing the project has to a rationale.

  1. Pitch carries the primary attribute. Pitch is the most commonly used
     ordered auditory channel, and it is quantized to a musical scale because
     absolute pitch judgement is rare while relative scale-degree judgement is
     common. Note the harder constraint: naming a value from a tone heard
     alone (absolute identification) saturates around 5-7 levels
     (Miller 1956, citing Pollack), which is far fewer than the number of
     pitches a scale can supply. `n_pitch_steps` exists to respect that.

  2. Direction is not universal. Walker (2002) found the preferred polarity
     depends on the data dimension -- rising pitch reads as rising
     temperature but as *falling* size -- and that polarity agreement predicts
     how well a mapping works. Hence the `polarity` argument rather than a
     hardcoded "higher is more".

  3. Brightness repeats the primary attribute. The evidence for redundancy
     gain is mixed: Peres & Lane (ICAD 2005) found a benefit only for
     *integral* dimension pairs (pitch+loudness) and none for separable ones,
     and several earlier studies found no benefit at all. Brightness is used
     here because it covaries with pitch perceptually -- it is closer to the
     integral case where a benefit has been observed -- and because it costs
     no extra channel. It is a hypothesis to test, not a settled gain.

  4. Categorical data goes to timbre. Timbre is NOT perceptually independent
     of pitch: pitch, loudness and timbre interact and are generally treated
     as integral (Melara & Marks; Grau & Kemler Nelson; Neuhoff). This means a
     timbre mapping can distort the pitch reading -- Neuhoff's stock-market
     study showed exactly that kind of "data lost in the map" effect. Timbre
     is still used because it is the only categorical channel available that
     survives mono playback, but a design that depends on reading pitch
     *precisely* while timbre varies should be checked with listeners, or
     timbre should be dropped via `categorical=False`.

  5. Channels compete for attention, so the default caps how many are used.
     How many streams a listener can follow depends heavily on the cues
     separating them, so the cap is a guide rather than a measured limit.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import numpy as np

from sensing_network.sonification.attributes import (
    node_attributes,
    describe_attribute,
    suggest_polarity,
    ATTRIBUTE_FUNCTIONS,
)
from sensing_network.sonification.encoding import (
    Channel,
    Encoding,
    SonificationSpec,
    ToneSpec,
    CHANNEL_SPECS,
)
from sensing_network.sonification.scales import (
    LinearScale,
    LogScale,
    OrdinalScale,
    QuantizedScale,
)


def _is_categorical(name, values):
    _, categorical, _ = describe_attribute(name)
    if categorical:
        return True
    values = np.asarray(values, dtype=float)
    # Treat a small number of distinct integral values as categorical.
    distinct = np.unique(values)
    return len(distinct) <= 6 and np.allclose(distinct, np.round(distinct))


# Thresholds for switching an attribute onto a log scale. These are
# engineering heuristics, not derived constants, and they change how data is
# represented -- so they are named, documented, overridable, and recorded in
# the spec's metadata. A reader of a generated figure can therefore see which
# scale was used and why, which a silent switch would not allow.
# Default number of distinct pitch levels. Absolute identification along a
# single auditory dimension runs out at about 5-7 categories (Miller 1956,
# citing Pollack's pitch data), and a listener here is also attending to touch,
# so the conservative end is taken. More levels are not *wrong* -- they still
# convey ordering -- but they stop conveying values, which is what a
# physicalization reader usually wants.
DEFAULT_PITCH_STEPS = 5

LOG_SCALE_DYNAMIC_RANGE = 20.0  # max/min ratio above which a log scale is considered
LOG_SCALE_SKEW_RATIO = 1.5  # mean/median ratio indicating a heavy right tail


def _should_use_log(values, dynamic_range=None, skew_ratio=None):
    """
    Whether an attribute is skewed enough to warrant a log scale.

    Heavy-tailed attributes (degree in a scale-free graph, say) are compressed
    into a few high pitches by a linear scale, wasting most of the range. Both
    conditions must hold: a wide spread of magnitudes *and* a right skew, so a
    merely wide-but-uniform attribute stays linear.
    """
    dynamic_range = (
        LOG_SCALE_DYNAMIC_RANGE if dynamic_range is None else dynamic_range
    )
    skew_ratio = LOG_SCALE_SKEW_RATIO if skew_ratio is None else skew_ratio

    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if len(positive) < 3:
        return False
    if positive.max() / max(positive.min(), 1e-12) < dynamic_range:
        return False
    # Compare mean to median: a large gap indicates a right skew.
    return float(np.mean(positive)) > skew_ratio * float(np.median(positive))


def _pitch_scale(values, pitch_set, root_midi, n_octaves, n_steps, polarity=1):
    scale = QuantizedScale(
        pitch_set=pitch_set,
        root_midi=root_midi,
        n_octaves=n_octaves,
        n_steps=n_steps,
        polarity=polarity,
    )
    if _should_use_log(values):
        # Quantize the log-transformed value by pre-warping the domain.
        values = np.asarray(values, dtype=float)
        shift = 0.0 if values.min() > 0 else 1.0 - values.min()
        logged = np.log(values + shift)
        scale.fit(logged)
        scale._log_shift = shift
        return scale, logged
    scale.fit(values)
    return scale, np.asarray(values, dtype=float)


def default_sonification(
    nodes,
    links,
    primary="degree",
    secondary=None,
    categorical=None,
    extra_attributes=None,
    link_weights=None,
    pitch_set="pentatonic",
    root_midi=60,
    n_octaves=2,
    n_pitch_steps="auto",
    redundant_brightness=True,
    max_channels=4,
    ensure_identity=True,
    identity_strategy="auto",
    polarity=None,
    metadata=None,
):
    """
    Build a SonificationSpec for a network with sensible defaults.

    Parameters
    ----------
    nodes, links : as used throughout the pipeline.
    primary : attribute name mapped to pitch. This should be the attribute the
        physicalization is chiefly about.
    secondary : optional attribute mapped to a second ordered channel
        (pulse rate), for a two-variable sonification.
    categorical : optional categorical attribute mapped to timbre. Defaults to
        'community' when the network is large enough for communities to mean
        something.
    extra_attributes : dict of {name: per-node values} for data not derived
        from graph structure.
    pitch_set, root_midi, n_octaves : the pitch collection (see scales).
    n_pitch_steps : cap on distinct pitches. "auto" (the default) caps at
        DEFAULT_PITCH_STEPS, because naming a value from a tone heard alone --
        the task a person touching a single node performs -- saturates at
        roughly 5-7 categories (Miller 1956, citing Pollack). Pass None to use
        the whole collection when the task is ranking rather than reading
        values, or an integer to choose deliberately.
    redundant_brightness : also map the primary attribute to brightness, which
        measurably improves ordered judgements.
    max_channels : upper bound on channels carrying *data*. An identity cue
        added by `ensure_identity` is not a data mapping and is not counted
        against this budget.
    ensure_identity : add an identity cue when the data mapping alone leaves
        nodes sounding identical (as it does on any regular graph, where every
        node shares a degree). Without this, a listener can hear a node's value
        but not tell which node it is.
    identity_strategy : "auto", "channel", or "prefix". See
        sensing_network.sonification.identity.
    polarity : +1 (higher data -> higher pitch) or -1 (inverted) for the
        primary attribute. Defaults to a per-attribute suggestion, falling back
        to +1. Walker (2002) showed the preferred direction depends on the data
        dimension -- rising pitch reads as rising temperature but as *falling*
        size -- and that polarity agreement predicts mapping effectiveness, so
        this is worth setting deliberately for unfamiliar attributes.

    Returns
    -------
    SonificationSpec
    """
    nodes = list(nodes)

    wanted = [primary]
    if secondary:
        wanted.append(secondary)

    if categorical is None and len(nodes) >= 6:
        categorical = "community"
    if categorical:
        wanted.append(categorical)

    # Only compute graph-derived attributes; anything else must come in via
    # extra_attributes.
    graph_attrs = [a for a in dict.fromkeys(wanted) if a in ATTRIBUTE_FUNCTIONS]
    data = node_attributes(
        nodes,
        links,
        attributes=graph_attrs,
        extra=extra_attributes,
        link_weights=link_weights,
    )

    for name in wanted:
        if name not in data:
            raise ValueError(
                f"attribute '{name}' is neither a built-in graph attribute nor "
                f"supplied in extra_attributes"
            )

    encoding = Encoding()

    # 1. Primary attribute -> pitch.
    if polarity is None:
        polarity = suggest_polarity(primary)
        polarity_source = "suggested" if polarity is not None else "default"
        if polarity is None:
            polarity = 1
    else:
        polarity_source = "explicit"

    if n_pitch_steps == "auto":
        # Never ask for more levels than there are nodes: a 3-node network
        # does not need 5 pitches.
        n_pitch_steps = min(DEFAULT_PITCH_STEPS, max(len(nodes), 2))

    pitch_values = data[primary]
    pitch_scale, pitch_input = _pitch_scale(
        pitch_values, pitch_set, root_midi, n_octaves, n_pitch_steps, polarity
    )
    encoding.add(Channel("pitch", primary, pitch_scale))
    # Keep the (possibly log-warped) input used for the pitch scale so the
    # spec's stored domain and the values agree.
    pitch_input_by_attr = {primary: pitch_input}

    # 2. Redundant brightness on the same attribute.
    if redundant_brightness and len(encoding.channels) < max_channels:
        bright = LinearScale(
            range_=CHANNEL_SPECS["brightness"]["default_range"], polarity=polarity
        )
        bright.fit(pitch_input)
        encoding.add(
            Channel(
                "brightness",
                primary,
                bright,
                meaning=describe_attribute(primary)[2],
            )
        )

    # 3. Categorical attribute -> timbre.
    if categorical and len(encoding.channels) < max_channels:
        values = data[categorical]
        labels = [int(v) if float(v).is_integer() else v for v in values]
        timbres = list(CHANNEL_SPECS["timbre"]["default_range"])
        ordinal = OrdinalScale(range_=timbres)
        ordinal.fit(labels)
        encoding.add(Channel("timbre", categorical, ordinal))
        data[categorical] = np.asarray(values, dtype=float)

    # 4. Secondary ordered attribute -> pulse rate. Rhythmic cues are the
    #    best available candidate for a second stream: Peres & Lane treated
    #    pitch and temporal rate as a *separable* pair, in contrast to the
    #    integral pitch/loudness pair. Separability is why it can carry an
    #    independent variable -- and also why it should not be expected to
    #    reinforce the pitch mapping.
    if secondary and len(encoding.channels) < max_channels:
        values = data[secondary]
        rng = CHANNEL_SPECS["pulse_rate"]["default_range"]
        scale = LogScale(range_=rng) if _should_use_log(values) else LinearScale(range_=rng)
        scale.fit(values)
        encoding.add(Channel("pulse_rate", secondary, scale))

    # Resolve every channel to per-node values.
    apply_input = dict(data)
    apply_input.update(pitch_input_by_attr)

    resolved = {}
    for ch in encoding.channels:
        source = (
            pitch_input_by_attr[primary]
            if (ch.attribute == primary and ch.name in ("pitch", "brightness"))
            else data[ch.attribute]
        )
        if ch.name == "timbre":
            labels = [int(v) if float(v).is_integer() else v for v in source]
            resolved[ch.name] = ch.scale(labels)
        else:
            out = ch.apply(source)
            resolved[ch.name] = out.tolist() if isinstance(out, np.ndarray) else list(out)

    tones = []
    for i, node in enumerate(nodes):
        params = {name: values[i] for name, values in resolved.items()}
        attrs = {name: float(values[i]) for name, values in data.items()}
        node_id = int(node) if isinstance(node, (int, np.integer)) else node
        tones.append(ToneSpec(node_id, params, attrs))

    meta = {
        "primary": primary,
        "secondary": secondary,
        "categorical": categorical,
        "pitch_set": pitch_set,
        "root_midi": root_midi,
        "n_octaves": n_octaves,
        "n_nodes": len(nodes),
        "polarity": polarity,
        "polarity_source": polarity_source,
        # Recorded so a generated sonification can be reproduced and audited:
        # the log switch is automatic and otherwise invisible.
        "pitch_scale_transform": (
            "log" if _should_use_log(data[primary]) else "linear"
        ),
        "log_thresholds": {
            "dynamic_range": LOG_SCALE_DYNAMIC_RANGE,
            "skew_ratio": LOG_SCALE_SKEW_RATIO,
        },
    }
    meta.update(metadata or {})

    spec = SonificationSpec(
        nodes=nodes,
        encoding=encoding,
        tones=tones,
        attribute_data=data,
        links=links,
        metadata=meta,
    )

    if ensure_identity:
        # Imported here to keep the module import graph acyclic: identity
        # builds on the encoding types this module also produces.
        from sensing_network.sonification.identity import ensure_identifiable

        spec, report = ensure_identifiable(spec, strategy=identity_strategy)
        spec.metadata["identity"] = {
            "action": report["action"],
            "reason": report["reason"],
            "fraction_identifiable": report["after"]["fraction_identifiable"],
            # Whether the cue actually fixed the ambiguity. A cue can be
            # applied and still leave nodes confusable (too many nodes for the
            # channel), and reporting the action alone would hide that.
            "resolved": report.get(
                "resolved", report["after"]["fraction_identifiable"] >= 1.0
            ),
        }

    return spec


def sonify_network(pipeline_result, outfile_path=None, **kwargs):
    """
    Build a sonification directly from a `default_pipeline` result.

    This is the one-call bridge between the fabrication pipeline and the audio
    layer: node IDs in the returned spec are exactly the node IDs the physical
    artifact senses, so calibration output can index into it directly.
    """
    spec = default_sonification(
        nodes=pipeline_result["nodes"],
        links=pipeline_result["links"],
        metadata={
            "in_node": pipeline_result.get("in_node"),
            "out_node": pipeline_result.get("out_node"),
            "node_radius": pipeline_result.get("node_radius"),
            "link_radius": pipeline_result.get("link_radius"),
        },
        **kwargs,
    )
    if outfile_path:
        spec.save(outfile_path)
    return spec
