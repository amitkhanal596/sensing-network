"""
Scales mapping data values to perceptual audio parameters.

The design mirrors visual encoding scales (cf. D3/Vega): a scale takes a
domain in data space and maps it to a range in a perceptual space. The
difference is that the perceptual spaces here are auditory, so the defaults
are chosen for auditory rather than visual discriminability.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import numpy as np


def midi_to_hz(midi):
    """
    Convert MIDI note numbers to frequencies in Hz (A4 = MIDI 69 = 440 Hz).
    """
    return 440.0 * np.power(2.0, (np.asarray(midi, dtype=float) - 69.0) / 12.0)


def hz_to_midi(hz):
    """
    Convert frequencies in Hz to MIDI note numbers.
    """
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0)


# Pitch collections used to quantize continuous values onto musical scales.
# Quantizing trades resolution for identifiability: listeners judge relative
# pitch far more reliably when the pitches belong to a familiar collection.
# Values are semitone offsets within an octave.
PITCH_SETS = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "whole_tone": [0, 2, 4, 6, 8, 10],
    # Octaves alone: maximally distinguishable, useful for ordinal data with
    # very few levels.
    "octaves": [0],
}


def _as_float_array(values):
    return np.asarray(values, dtype=float)


class Scale:
    """
    Base class. Subclasses implement __call__ and to_dict.
    """

    kind = "scale"

    def __call__(self, values):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.to_dict()})"


class LinearScale(Scale):
    """
    Linear map from `domain` to `range_`.

    Values outside the domain are clamped by default, so that an outlier can
    never produce an inaudible or painful output.

    `polarity` of -1 inverts the mapping. This is not cosmetic: Walker (2002)
    found that the preferred direction depends on the data dimension --
    listeners map rising pitch to rising temperature but to *falling* size --
    and that polarity agreement predicts how well a mapping works.
    """

    kind = "linear"

    def __init__(self, domain=None, range_=(0.0, 1.0), clamp=True, polarity=1):
        self.domain = None if domain is None else tuple(float(d) for d in domain)
        self.range_ = tuple(float(r) for r in range_)
        self.clamp = clamp
        self.polarity = -1 if polarity < 0 else 1

    def fit(self, values):
        values = _as_float_array(values)
        lo, hi = float(np.min(values)), float(np.max(values))
        if lo == hi:
            # A constant attribute carries no information; center it in the
            # range rather than producing a degenerate (0-width) domain.
            lo, hi = lo - 0.5, hi + 0.5
        self.domain = (lo, hi)
        return self

    def __call__(self, values):
        if self.domain is None:
            self.fit(values)
        values = _as_float_array(values)
        lo, hi = self.domain
        t = (values - lo) / (hi - lo)
        if self.clamp:
            t = np.clip(t, 0.0, 1.0)
        if self.polarity < 0:
            t = 1.0 - t
        r0, r1 = self.range_
        return r0 + t * (r1 - r0)

    def to_dict(self):
        return {
            "type": self.kind,
            "domain": list(self.domain) if self.domain else None,
            "range": list(self.range_),
            "clamp": self.clamp,
            "polarity": self.polarity,
        }


class LogScale(Scale):
    """
    Logarithmic map, for attributes spanning orders of magnitude (e.g. degree
    in a scale-free network, edge weights, populations).

    Non-positive values are shifted so the transform stays defined.
    """

    kind = "log"

    def __init__(self, domain=None, range_=(0.0, 1.0), clamp=True, base=10.0, polarity=1):
        self.domain = None if domain is None else tuple(float(d) for d in domain)
        self.range_ = tuple(float(r) for r in range_)
        self.clamp = clamp
        self.base = float(base)
        self._shift = 0.0
        self.polarity = -1 if polarity < 0 else 1

    def _prepare(self, values):
        values = _as_float_array(values)
        min_val = float(np.min(values)) if values.size else 0.0
        # Shift so the smallest value maps to 1.0 under the log.
        self._shift = 0.0 if min_val > 0 else (1.0 - min_val)
        return values + self._shift

    def fit(self, values):
        shifted = self._prepare(values)
        lo, hi = float(np.min(shifted)), float(np.max(shifted))
        if lo == hi:
            hi = lo * self.base
        self.domain = (lo, hi)
        return self

    def __call__(self, values):
        if self.domain is None:
            self.fit(values)
        values = _as_float_array(values) + self._shift
        values = np.maximum(values, 1e-12)
        lo, hi = self.domain
        lo, hi = max(lo, 1e-12), max(hi, 1e-12)
        t = (np.log(values) - np.log(lo)) / (np.log(hi) - np.log(lo))
        if self.clamp:
            t = np.clip(t, 0.0, 1.0)
        if self.polarity < 0:
            t = 1.0 - t
        r0, r1 = self.range_
        return r0 + t * (r1 - r0)

    def to_dict(self):
        return {
            "type": self.kind,
            "domain": list(self.domain) if self.domain else None,
            "range": list(self.range_),
            "clamp": self.clamp,
            "base": self.base,
            "shift": self._shift,
            "polarity": self.polarity,
        }


class OrdinalScale(Scale):
    """
    Map discrete categories to a list of output values (e.g. timbres).
    """

    kind = "ordinal"

    def __init__(self, domain=None, range_=None):
        self.domain = list(domain) if domain is not None else None
        self.range_ = list(range_) if range_ is not None else []

    def fit(self, values):
        seen = []
        for v in values:
            if v not in seen:
                seen.append(v)
        self.domain = seen
        return self

    def __call__(self, values):
        if self.domain is None:
            self.fit(values)
        if not self.range_:
            raise ValueError("OrdinalScale requires a non-empty range")
        out = []
        for v in values:
            idx = self.domain.index(v) if v in self.domain else 0
            out.append(self.range_[idx % len(self.range_)])
        return out

    def to_dict(self):
        return {
            "type": self.kind,
            "domain": list(self.domain) if self.domain else None,
            "range": list(self.range_),
        }


class QuantizedScale(Scale):
    """
    Map continuous values onto a musical pitch collection.

    This is the workhorse for pitch encoding. Rather than mapping a value to
    an arbitrary frequency, it maps to the nearest degree of a named pitch set
    spanning `n_octaves` from `root_midi`. The result is returned in Hz.

    `n_steps` optionally limits how many distinct pitches are used, which
    directly controls how many levels a listener is asked to distinguish.
    """

    kind = "quantized"

    def __init__(
        self,
        domain=None,
        pitch_set="pentatonic",
        root_midi=60,
        n_octaves=2,
        n_steps=None,
        clamp=True,
        polarity=1,
    ):
        self.domain = None if domain is None else tuple(float(d) for d in domain)
        self.polarity = -1 if polarity < 0 else 1
        self.pitch_set = pitch_set
        self.root_midi = int(root_midi)
        self.n_octaves = int(n_octaves)
        self.n_steps = n_steps
        self.clamp = clamp

    def pitches_midi(self):
        """
        The MIDI notes available in this scale, ascending.
        """
        offsets = PITCH_SETS.get(self.pitch_set, PITCH_SETS["pentatonic"])
        notes = []
        for octave in range(self.n_octaves + 1):
            for off in offsets:
                notes.append(self.root_midi + 12 * octave + off)
        notes = sorted(set(notes))
        # Cap the top of the range at exactly n_octaves above the root.
        notes = [n for n in notes if n <= self.root_midi + 12 * self.n_octaves]
        if self.n_steps is not None and self.n_steps > 0:
            n = min(int(self.n_steps), len(notes))
            if n == 1:
                notes = [notes[0]]
            else:
                # Spread the requested number of steps evenly across the range.
                idx = np.linspace(0, len(notes) - 1, n)
                notes = [notes[int(round(i))] for i in idx]
        return notes

    def fit(self, values):
        values = _as_float_array(values)
        lo, hi = float(np.min(values)), float(np.max(values))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        self.domain = (lo, hi)
        return self

    def __call__(self, values):
        if self.domain is None:
            self.fit(values)
        values = _as_float_array(values)
        lo, hi = self.domain
        t = (values - lo) / (hi - lo)
        if self.clamp:
            t = np.clip(t, 0.0, 1.0)
        if self.polarity < 0:
            t = 1.0 - t
        notes = self.pitches_midi()
        idx = np.round(t * (len(notes) - 1)).astype(int)
        idx = np.clip(idx, 0, len(notes) - 1)
        midi = np.array([notes[i] for i in idx], dtype=float)
        return midi_to_hz(midi)

    def to_dict(self):
        return {
            "type": self.kind,
            "domain": list(self.domain) if self.domain else None,
            "pitch_set": self.pitch_set,
            "root_midi": self.root_midi,
            "n_octaves": self.n_octaves,
            "n_steps": self.n_steps,
            "clamp": self.clamp,
            "polarity": self.polarity,
            "pitches_midi": self.pitches_midi(),
        }


def scale_from_dict(d):
    """
    Rebuild a scale from its dict form (the inverse of to_dict).
    """
    kind = d.get("type")
    if kind == "linear":
        return LinearScale(
            domain=d.get("domain"),
            range_=d.get("range", (0, 1)),
            clamp=d.get("clamp", True),
            polarity=d.get("polarity", 1),
        )
    if kind == "log":
        s = LogScale(
            domain=d.get("domain"),
            range_=d.get("range", (0, 1)),
            clamp=d.get("clamp", True),
            base=d.get("base", 10.0),
            polarity=d.get("polarity", 1),
        )
        s._shift = d.get("shift", 0.0)
        return s
    if kind == "ordinal":
        return OrdinalScale(domain=d.get("domain"), range_=d.get("range"))
    if kind == "quantized":
        return QuantizedScale(
            domain=d.get("domain"),
            pitch_set=d.get("pitch_set", "pentatonic"),
            root_midi=d.get("root_midi", 60),
            n_octaves=d.get("n_octaves", 2),
            n_steps=d.get("n_steps"),
            clamp=d.get("clamp", True),
            polarity=d.get("polarity", 1),
        )
    raise ValueError(f"unknown scale type: {kind}")
