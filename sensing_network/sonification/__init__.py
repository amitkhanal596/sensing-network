"""
Sonification layer for sensing network physicalizations.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

from sensing_network.sonification.scales import (
    LinearScale,
    LogScale,
    OrdinalScale,
    QuantizedScale,
    PITCH_SETS,
    midi_to_hz,
    hz_to_midi,
)
from sensing_network.sonification.encoding import (
    Channel,
    Encoding,
    SonificationSpec,
    ToneSpec,
)
from sensing_network.sonification.attributes import (
    node_attributes,
    ATTRIBUTE_FUNCTIONS,
    describe_attribute,
)
from sensing_network.sonification.pipeline import (
    default_sonification,
    sonify_network,
)
from sensing_network.sonification.identity import (
    ensure_identifiable,
    identity_ambiguity,
    perceptual_distinctness,
    add_identity_channel,
    add_identity_prefix,
)
from sensing_network.sonification.forms import (
    line_graph_form,
    bar_chart_form,
    scatter_form,
    region_adjacency_form,
    sonify_form,
)

__all__ = [
    "LinearScale",
    "LogScale",
    "OrdinalScale",
    "QuantizedScale",
    "PITCH_SETS",
    "midi_to_hz",
    "hz_to_midi",
    "Channel",
    "Encoding",
    "SonificationSpec",
    "ToneSpec",
    "node_attributes",
    "ATTRIBUTE_FUNCTIONS",
    "describe_attribute",
    "default_sonification",
    "sonify_network",
    "ensure_identifiable",
    "identity_ambiguity",
    "perceptual_distinctness",
    "add_identity_channel",
    "add_identity_prefix",
    "line_graph_form",
    "bar_chart_form",
    "scatter_form",
    "region_adjacency_form",
    "sonify_form",
]
