# Multimodal Physicalization: Sonification Pipeline

An extension to the sensing network physicalization pipeline that makes a
printed object *speak its data* when touched. The fabrication pipeline is
unchanged; this adds the layer that turns a sensed touch into sound.

## Why

A sensing physicalization already gives a reader shape they can feel and a
touch the hardware can detect. What it does not give them is the value. A
reader can tell that one node is larger, but not that its betweenness is 0.57;
they can trace a link, but not hear how the network's structure differs around
it. Sound closes that gap, and it does so through a channel that does not
compete with touch for the reader's hands or with vision for their eyes.

## What is here

| Piece | Path | What it does |
| --- | --- | --- |
| Sonification module | `sensing_network/sonification/` | Turns data into a portable sound specification |
| Audio engine | `callibration/public/js/sonification.js` | Plays a specification in any browser |
| Touch classifier | `callibration/public/js/classifier.js` | Turns noisy sensor readings into confident node IDs |
| Explorer UI | `callibration/public/explore.html` | Touch a node, hear and read its data |
| One-shot pipeline | `sonify_pipeline.py` | Network in, STLs + sonification out |
| Design checker | `check_design.py` | Catches unlistenable designs before you print |
| Tests | `tests/` | 33 Python + 18 JavaScript |

## Quick start

```bash
# Generate printable geometry and a matching sonification
python sonify_pipeline.py --network usage_scenario --primary degree --secondary betweenness

# Check the design is actually listenable before printing
python check_design.py result/sonification.json

# Play it: serve the calibration app, then open /explore
cd callibration && npm install && npm start
```

The explorer works with no hardware attached, so a sonification can be
designed and evaluated before anything is printed.

## How the mapping is chosen

The pipeline picks an encoding automatically. The rules below are **design
defaults informed by the sonification literature, not validated findings for
this hardware**; where the evidence is contested that is said plainly, because
these defaults are the closest thing the project has to a rationale and they
should not be cited as more than they are.

- **Pitch carries the primary attribute**, quantized onto a musical scale
  (pentatonic by default) because absolute pitch is rare while relative
  scale-degree judgement is common. The binding constraint is not pitch
  *discrimination* (telling two tones apart, possible to a fraction of a
  semitone) but *identification* — naming a value from a tone heard alone,
  which is what touching a single node requires. That saturates at roughly
  5–7 categories (Miller 1956, citing Pollack), so `n_pitch_steps` defaults
  to 5. Pass `None` to lift the cap when the task is ranking, not reading.
- **Direction is not universal.** Walker (2002) found the preferred polarity
  depends on the data dimension: rising pitch reads as rising *temperature*
  but as *falling* size. Hence a `polarity` argument, with size-like
  attributes defaulting to negative. The legend states the direction actually
  used.
- **Brightness repeats the primary attribute.** This is a hypothesis, not a
  settled gain: Peres & Lane (ICAD 2005) found redundancy helped only for
  *integral* dimension pairs and not for separable ones, and earlier studies
  found no benefit at all. Brightness is used because it covaries with pitch
  perceptually — closer to the integral case where a benefit was observed —
  and because it costs no extra channel.
- **Timbre carries categories, with a caveat.** Timbre is *not* perceptually
  independent of pitch: pitch, loudness and timbre interact and are generally
  treated as integral (Melara & Marks; Grau & Kemler Nelson; Neuhoff). A
  timbre mapping can therefore distort the pitch reading — Neuhoff's
  stock-market study showed exactly that "data lost in the map" effect. Pass
  `categorical=False` if precise pitch reading matters more than category.
- **Pulse rate carries a second ordered attribute.** Pitch and temporal rate
  are treated as a separable pair, which is why rate can carry an independent
  variable — and equally why it should not be expected to reinforce pitch.
- **Heavy-tailed attributes get a log scale automatically.** The thresholds
  are named constants (`LOG_SCALE_DYNAMIC_RANGE`, `LOG_SCALE_SKEW_RATIO`),
  and which transform was applied is recorded in the spec's metadata, so a
  generated figure can be audited rather than silently reinterpreted.

Every choice lands in `sonification.json` alongside a plain-language legend,
so the mapping can be read, checked, and explained — including by a screen
reader.

## The identity problem

A data-driven sonification has a failure mode that a visual encoding does not,
and it took building the thing to see it clearly.

On a complete graph every node has the same degree, so a degree-to-pitch
mapping gives every node **the same sound**. A sighted reader resolves this by
glancing at the object. A blind reader cannot — which breaks the sonification
precisely for the audience it was built to serve.

`sensing_network/sonification/identity.py` detects this and fixes it, without
disturbing the data mapping:

- **Identity prefix** (the default): a short, quiet marker tone before the
  data tone, in a higher register so it reads as punctuation rather than
  measurement. It costs time instead of a channel, leaves the data tone
  untouched, works on any output device, and scales — markers stay distinct
  up to 96 nodes, the largest network the original paper fabricates.
- **Identity channel**: each node gets a distinct value on an unused channel.
  Used automatically only when the channel can carry the node count with
  levels a listener can actually *identify* (see `CHANNEL_IDENTIFY_CAPACITY`),
  which for four waveforms means four nodes. Stereo pan is available but is
  deliberately not the automatic default: it is inaudible on mono output — one
  earbud, a laptop or phone speaker — which is a common setup for a listener
  keeping one ear on the room. The audio engine detects mono output, warns,
  and substitutes a pitched marker.

**Judging ambiguity honestly.** `perceptual_distinctness()` counts two tones
as different only when at least one channel differs by more than that
channel's perceptual resolution. The older `identity_ambiguity()` compares
floating-point values and is kept only for checking that the encoding assigns
different numbers — it will happily call 20 stereo positions spaced 0.08 apart
"distinct", which no listener can identify. The resolutions are engineering
approximations chosen to err toward "these may sound the same"; validating
them needs a listening study.

`check_design.py` reports remaining ambiguity, pitch spacing *and* pitch level
count (the two different psychophysical limits), automatic scale transforms,
and channels wasted on near-constant attributes.

## Sensing reliability

Sound makes classification errors obvious in a way a text label never did: a
wrong node is a wrong note, and flicker between two nodes is an audible mess.
The classifier was rebuilt accordingly:

- Each node is modelled with a **mean and a spread**, so a noisy node stops
  stealing readings from its tighter neighbours.
- Readings that fit nothing are **rejected rather than guessed** — silence
  beats a confident wrong note. "Between two nodes" and "outside the
  calibrated range" are reported separately, because they call for different
  fixes.
- A **median pre-filter** removes the isolated spikes typical of capacitive
  sensing.
- **Baseline drift** is tracked, addressing the "improving calibration"
  limitation the original paper names as future work. The correction is
  bounded and only learns from readings that still look like the baseline, so
  a hand left resting on the object or a loose wire cannot drag it away — an
  unbounded version deafened the device permanently in testing.
- **Which drift model applies is measured, not assumed.** A baseline shift may
  propagate additively (same offset everywhere) or multiplicatively (RC delay
  scales with resistance). The default is the conservative additive model;
  `observeKnownTouch()` plus `driftModelFit()` identify which model a given
  rig follows from a handful of labelled touches, and applying the detected
  model restored full accuracy under both physics in simulation.
- **Data that breaks the model is flagged.** A node touched two different ways
  produces two clusters, and a single Gaussian then places its mean in a gap
  where it never reads — looking well separated while classifying badly. Each
  node's samples are screened for bimodality and the separation report says
  when its own figures should not be trusted.
- A **stability tracker** requires agreement before committing to a node, and
  a shorter agreement to release, so letting go still feels immediate.
- **Separation is reported in sigmas** during calibration, so a physicalization
  that cannot be sensed reliably is caught before it is printed rather than
  after.

The richer calibration format is backward compatible: existing `config.json`
files still load, with a per-node spread estimated from the spacing of the
node means.

## Beyond networks

`sensing_network/sonification/forms.py` generalizes the approach. The sensing
pipeline needs "touchable regions connected in a chain or mesh", which many
visualizations already are:

- `line_graph_form` — a path graph; the shape can be traced by hand while
  sound carries exact values
- `bar_chart_form` — bar tops as touch targets
- `scatter_form` — points joined by a minimum spanning tree (least added
  material and least clutter) or k-nearest neighbours, always stitched into
  one component because the sensing chain requires it
- `region_adjacency_form` — choropleth-style regions with adjacency links

For the spatial forms the links are **structural, not semantic**. Worth stating
in any caption: a reader tracing them is following fabrication, not data.

## Interaction modes

- **Discrete** — one tone per touch. Good for reading values point by point.
- **Continuous** — a sustained tone that changes as the hand moves, turning
  the surface into something closer to an instrument.
- **Tour** — plays every node in sequence, giving the overview a static object
  cannot.
- **Neighbourhood** — plays a node then its neighbours, so local structure can
  be heard without tracing it by hand.
- **Speech** — speaks the exact values, for when a number is wanted rather
  than a comparison.

## Testing

```bash
python tests/test_sonification.py   # 45 tests: scales, encodings, polarity,
                                    #   perceptual distinctness, forms
node tests/test_classifier.mjs      # 28 tests: classification, drift safety,
                                    #   drift-model detection, bimodality
node tests/test_audio.mjs           # 13 tests: audio scheduling, mono
                                    #   fallback, voice lifecycle
```

The audio tests use a stub audio context that records what the engine
schedules, so the control logic (marker/tone ordering, mono fallback, voices
always stopped) is testable in CI. Signal-level behaviour was verified
separately in a real browser: data tones measured within 2 Hz of their
specified frequencies, and a spectrogram walk confirmed the identity marker
completes before the data tone begins (1324 Hz → 586 Hz → 258 Hz, cleanly
separated).

**Still untested against real hardware.** Every classifier test uses synthetic
noise, which is the distribution the model assumes — so the tests cannot fail
for the reason the model is most likely wrong. The drift models in particular
are simulated, not measured on a printed network.

## What this does and does not claim

The defensible contribution is a pipeline that attaches a declarative,
inspectable sonification to fabricated sensing physicalizations, plus the
observation that structure-driven audio encodings collapse node identity on
regular graphs — a failure that specifically disadvantages non-visual users,
who cannot glance at the object to disambiguate.

Everything about *which* mapping is best is design rationale awaiting test,
not a finding of this work. The perceptual constants (pitch level caps,
channel capacities, resolution thresholds) are conservative engineering
choices grounded in published limits, not measurements on this hardware or
with these listeners.

## Suggested directions

Ordered by how much they would teach relative to effort.

1. **A listening study.** The design rests on claims that should be tested
   with real listeners: whether pitch plus brightness beats pitch alone (the
   redundancy literature is split), how many levels are reliably identifiable
   on a printed object, whether the timbre mapping distorts pitch reading as
   the integrality literature predicts, and whether a marker or a channel
   works better for locating a node. This is the most publishable piece and
   needs no new fabrication.
2. **Validate the drift model on hardware.** `driftModelFit()` can identify
   whether a rig drifts additively or multiplicatively from a few labelled
   touches. Running that on a printed network over a session would settle a
   question currently answered only in simulation, and feeds directly into the
   "improving calibration" future work the original paper names.
3. **Sonify links, not just nodes.** The current design is node-centric.
   Dragging a finger along a link could sweep between its endpoints' tones,
   making an edge audible as a transition rather than a gap.
4. **Multi-touch.** The present circuit senses one node at a time. Two
   simultaneous tones would let a reader compare two values directly, which is
   the operation sonification supports worst today.
5. **Print the mapping into the object.** The legend currently lives on a
   screen. A tactile or braille legend on the base would make the artifact
   self-contained.
6. **Close the loop from sensing to design.** `separationReport` already knows
   which nodes are hard to distinguish; that signal could feed back into the
   resistance optimization to space those nodes further apart on the next
   print.
