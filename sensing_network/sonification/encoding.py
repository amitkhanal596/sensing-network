"""
Declarative encodings from data attributes to sound parameters.

A `SonificationSpec` is a portable description of how a physicalization should
sound. It is deliberately data, not code: the Python pipeline produces it, the
web audio engine consumes it, and a person can read it to check what a sound
is claiming about the data. This mirrors how a visualization grammar separates
an encoding from its renderer.

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

import numpy as np

from sensing_network.sonification.scales import (
    LinearScale,
    LogScale,
    OrdinalScale,
    QuantizedScale,
    midi_to_hz,
    scale_from_dict,
)
from sensing_network.sonification.attributes import describe_attribute


# Audio channels a data attribute can be mapped to, with the range that keeps
# them usable. These bounds matter: pitch outside roughly 200-2000 Hz loses
# resolution for most listeners, and a brightness floor keeps quiet tones from
# vanishing on small speakers.
CHANNEL_SPECS = {
    "pitch": {
        "unit": "hz",
        # C4 to C6, computed rather than written out so the bound agrees
        # exactly with the frequencies QuantizedScale produces.
        "default_range": (float(midi_to_hz(60)), float(midi_to_hz(84))),
        "description": "fundamental frequency of the tone",
        "ordered": True,
    },
    "brightness": {
        "unit": "normalized",
        "default_range": (0.15, 1.0),
        "description": "spectral brightness (low-pass cutoff)",
        "ordered": True,
    },
    "loudness": {
        "unit": "normalized",
        "default_range": (0.35, 1.0),
        "description": "peak amplitude of the tone",
        "ordered": True,
    },
    "duration": {
        "unit": "seconds",
        "default_range": (0.18, 0.7),
        "description": "length of the tone",
        "ordered": True,
    },
    "pulse_rate": {
        "unit": "hz",
        "default_range": (0.0, 12.0),
        "description": "amplitude modulation rate (perceived as roughness/beating)",
        "ordered": True,
    },
    "tremolo_depth": {
        "unit": "normalized",
        "default_range": (0.0, 0.8),
        "description": "depth of amplitude modulation",
        "ordered": True,
    },
    "detune": {
        "unit": "cents",
        "default_range": (0.0, 30.0),
        "description": "detuning between stacked oscillators (thickness)",
        "ordered": True,
    },
    "pan": {
        "unit": "normalized",
        "default_range": (-0.8, 0.8),
        "description": "stereo position",
        "ordered": True,
    },
    "timbre": {
        "unit": "categorical",
        "default_range": ("sine", "triangle", "square", "sawtooth"),
        "description": "oscillator waveform",
        "ordered": False,
    },
    "attack": {
        "unit": "seconds",
        "default_range": (0.005, 0.12),
        "description": "onset time (percussive to soft)",
        "ordered": True,
    },
}


class Channel:
    """
    A single audio channel bound to a data attribute through a scale.
    """

    def __init__(self, name, attribute, scale, label=None, meaning=None):
        if name not in CHANNEL_SPECS:
            raise ValueError(
                f"unknown channel '{name}'. Available: {sorted(CHANNEL_SPECS)}"
            )
        self.name = name
        self.attribute = attribute
        self.scale = scale
        auto_label, _, auto_meaning = describe_attribute(attribute)
        self.label = label or auto_label
        self.meaning = meaning or auto_meaning

    def apply(self, values):
        return self.scale(values)

    def to_dict(self):
        return {
            "channel": self.name,
            "attribute": self.attribute,
            "label": self.label,
            "meaning": self.meaning,
            "unit": CHANNEL_SPECS[self.name]["unit"],
            "scale": self.scale.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        return Channel(
            name=d["channel"],
            attribute=d["attribute"],
            scale=scale_from_dict(d["scale"]),
            label=d.get("label"),
            meaning=d.get("meaning"),
        )

    def __repr__(self):
        return f"Channel({self.name} <- {self.attribute})"


class Encoding:
    """
    An ordered collection of channels, applied together to attribute data.
    """

    def __init__(self, channels=None):
        self.channels = list(channels) if channels else []

    def add(self, channel):
        self.channels.append(channel)
        return self

    def apply(self, attribute_data):
        """
        Compute per-node sound parameters.

        Returns a dict of {channel_name: list of per-node values}.
        """
        out = {}
        for ch in self.channels:
            if ch.attribute not in attribute_data:
                raise KeyError(
                    f"channel '{ch.name}' needs attribute '{ch.attribute}', "
                    f"which was not computed"
                )
            values = attribute_data[ch.attribute]
            result = ch.apply(values)
            out[ch.name] = (
                result.tolist() if isinstance(result, np.ndarray) else list(result)
            )
        return out

    def to_dict(self):
        return [ch.to_dict() for ch in self.channels]

    @staticmethod
    def from_dict(lst):
        return Encoding([Channel.from_dict(d) for d in lst])

    def __repr__(self):
        return f"Encoding({[c.name for c in self.channels]})"


class ToneSpec:
    """
    The concrete sound for one node: every channel resolved to a value.
    """

    def __init__(self, node_id, params, attributes=None):
        self.node_id = node_id
        self.params = dict(params)
        self.attributes = dict(attributes or {})

    def to_dict(self):
        return {
            "node": self.node_id,
            "params": self.params,
            "attributes": self.attributes,
        }

    def __repr__(self):
        return f"ToneSpec(node={self.node_id}, {self.params})"


class SonificationSpec:
    """
    The full, portable description of how a physicalization sounds.

    Contains the encoding (how data maps to sound), the resolved per-node
    tones (so a client needs no data-processing code), and a text legend
    (so the mapping is itself accessible, including to a screen reader).
    """

    def __init__(
        self,
        nodes,
        encoding,
        tones,
        attribute_data=None,
        links=None,
        metadata=None,
    ):
        self.nodes = list(nodes)
        self.encoding = encoding
        self.tones = list(tones)
        self.attribute_data = attribute_data or {}
        self.links = [list(l) for l in links] if links is not None else []
        self.metadata = dict(metadata or {})

    def tone_for(self, node_id):
        for tone in self.tones:
            if tone.node_id == node_id:
                return tone
        return None

    def legend(self):
        """
        A plain-language description of the mapping, one line per channel.
        """
        lines = []
        for ch in self.encoding.channels:
            spec = CHANNEL_SPECS[ch.name]
            if ch.attribute == "node_id":
                # An identity channel says which node, not how much of
                # anything, so the ordered phrasing below would misdescribe it.
                lines.append(
                    f"{ch.name.replace('_', ' ').capitalize()} identifies which "
                    f"node is touched; it does not encode data."
                )
            elif spec["ordered"]:
                scale_info = ch.scale.to_dict()
                rng = scale_info.get("range")
                # State the direction the scale actually uses. A legend that
                # always says "higher means more" is wrong whenever polarity
                # is inverted, and an inverted mapping is exactly the case a
                # listener most needs told.
                inverted = scale_info.get("polarity", 1) < 0
                if ch.name == "pitch":
                    direction = "lower" if inverted else "higher"
                    lines.append(
                        f"Pitch encodes {ch.label}: {direction} pitch means "
                        f"{ch.meaning}."
                    )
                elif rng:
                    readable = ch.name.replace("_", " ")
                    direction = "less" if inverted else "more"
                    lines.append(
                        f"{readable.capitalize()} encodes {ch.label}: "
                        f"{direction} {readable} means {ch.meaning}."
                    )
                else:
                    lines.append(f"{ch.name} encodes {ch.label}.")
            else:
                lines.append(
                    f"{ch.name.capitalize()} encodes {ch.label} "
                    f"({spec['description']})."
                )
        return lines

    def describe_node(self, node_id):
        """
        A spoken-language summary of one node, for a verbal readout mode.
        """
        tone = self.tone_for(node_id)
        if tone is None:
            return f"Node {node_id} is not in this sonification."
        parts = [f"Node {node_id}."]
        described = set()
        for ch in self.encoding.channels:
            # Several channels may share one attribute (e.g. redundant
            # brightness); say each fact once.
            if ch.attribute in described:
                continue
            value = tone.attributes.get(ch.attribute)
            if value is None:
                continue
            described.add(ch.attribute)
            if abs(value - round(value)) < 1e-9:
                value_text = f"{int(round(value))}"
            else:
                value_text = f"{value:.3g}"
            parts.append(f"{ch.label} {value_text}.")
        return " ".join(parts)

    def to_dict(self):
        return {
            "version": 1,
            "nodes": [int(n) if isinstance(n, (int, np.integer)) else n for n in self.nodes],
            "links": self.links,
            "encoding": self.encoding.to_dict(),
            "tones": [t.to_dict() for t in self.tones],
            "attributes": {
                k: (v.tolist() if isinstance(v, np.ndarray) else list(v))
                for k, v in self.attribute_data.items()
            },
            "legend": self.legend(),
            "metadata": self.metadata,
        }

    def save(self, outfile_path):
        directory = os.path.dirname(outfile_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(outfile_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return outfile_path

    @staticmethod
    def from_dict(d):
        encoding = Encoding.from_dict(d["encoding"])
        tones = [
            ToneSpec(t["node"], t["params"], t.get("attributes")) for t in d["tones"]
        ]
        return SonificationSpec(
            nodes=d["nodes"],
            encoding=encoding,
            tones=tones,
            attribute_data=d.get("attributes"),
            links=d.get("links"),
            metadata=d.get("metadata"),
        )

    @staticmethod
    def load(path):
        with open(path) as f:
            return SonificationSpec.from_dict(json.load(f))

    def __repr__(self):
        return (
            f"SonificationSpec({len(self.nodes)} nodes, "
            f"channels={[c.name for c in self.encoding.channels]})"
        )
