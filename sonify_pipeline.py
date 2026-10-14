"""
End-to-end pipeline: a network in, printable geometry and a sonification out.

This is the single entry point the project needs. It runs the existing
fabrication pipeline (layout, resistor selection, resistance optimization,
path generation, STL export) and then attaches a sonification whose node IDs
are exactly the node IDs the finished object will sense -- so nothing has to
be lined up by hand between the printed artifact and the audio.

Usage:
    python sonify_pipeline.py --network usage_scenario --out result/
    python sonify_pipeline.py --form line --values 3,7,2,9,5 --out result/
    python sonify_pipeline.py --network simple --primary betweenness --fast

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import argparse
import json
import os
import shutil
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sensing_network.pipeline import default_pipeline
from sensing_network.convert_utils import (
    output_to_json,
    output_to_stl,
    output_to_3dforce_json,
)
from sensing_network.sonification import sonify_network
from sensing_network.sonification.forms import (
    line_graph_form,
    bar_chart_form,
    scatter_form,
    sonify_form,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import sample_networks


# Reduced settings for a quick design iteration. The optimizer and the layout
# adjustment dominate runtime, and neither changes the sonification, so a
# draft run can cut both while still producing a usable preview.
FAST_KWARGS = {
    "resistance_optimization_kwargs": {
        "buget_for_single_iter": 300,
        "max_iterations": 500,
        "convergence_thres": 0.05,
        "convergence_judge_buffer_size": 100,
    },
    "layout_adjustment_kwargs": {
        "layout_learning_epoch": 30,
        "main_learning_epoch": 30,
        "loss_weights": {
            "layout_change": 1,
            "prior_link_intersect": 3,
            "link_intersect": 1,
            "node_intersect": 3,
            "link_length_variation": 1,
        },
    },
}


def build_network(nodes, links, out_dir, fast=False, skip_stl=False, **son_kwargs):
    """
    Run the fabrication pipeline and the sonification over the same network.
    """
    pipeline_kwargs = dict(
        node_radius=10.0,
        link_radius=3.9,
        mean_link_length=50,
        resistance_range=[20.0e3, 300.0e3],
        resistor_path_generation_kwargs={
            "vertical_step": 1.0,
            "path_margin": 0.65,
            "vertical_box_width": 1.2,
            "vertical_box_additional_height": 0.5,
            "min_node_link_overlap": 3.0,
            "vertical_resistivity": 930.0,
            "horizontal_resistivity": 570.0,
        },
        use_out_node=True,
        verbose=True,
    )
    if fast:
        pipeline_kwargs.update(FAST_KWARGS)

    print("running fabrication pipeline...")
    result = default_pipeline(nodes=nodes, links=links, **pipeline_kwargs)

    os.makedirs(out_dir, exist_ok=True)

    output_to_json(outfile_path=os.path.join(out_dir, "cad_data.json"), **result)
    output_to_3dforce_json(
        nodes,
        links,
        node_positions=result["node_positions"],
        node_radius=result["node_radius"],
        link_radius=result["link_radius"],
        outfile_path=os.path.join(out_dir, "nw.json"),
    )
    if not skip_stl:
        print("writing STL files...")
        output_to_stl(outfile_path=os.path.join(out_dir, "stl_data"), **result)

    print("building sonification...")
    spec = sonify_network(
        result, outfile_path=os.path.join(out_dir, "sonification.json"), **son_kwargs
    )
    return result, spec


def build_form(form, out_dir, **son_kwargs):
    """
    Sonify a non-network form. The geometry is described by the form itself,
    so this writes the form and its sonification without running the resistor
    optimization -- useful for designing the audio before committing to a print.
    """
    os.makedirs(out_dir, exist_ok=True)
    spec = sonify_form(form, **son_kwargs)
    spec.save(os.path.join(out_dir, "sonification.json"))

    with open(os.path.join(out_dir, "form.json"), "w") as f:
        json.dump(
            {
                "form": form["form"],
                "nodes": form["nodes"],
                "links": form["links"],
                "node_positions": np.asarray(form["node_positions"]).tolist(),
                "metadata": form["metadata"],
            },
            f,
            indent=2,
        )
    return spec


def publish_to_calibration(out_dir):
    """
    Copy the sonification where the web explorer expects to find it.
    """
    target_dir = os.path.join("callibration", "public", "data")
    source = os.path.join(out_dir, "sonification.json")
    if not os.path.exists(source):
        return None
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "sonification.json")
    shutil.copy(source, target)
    return target


def summarize(spec, result=None):
    print("\n" + "=" * 62)
    print("SONIFICATION")
    print("=" * 62)
    for line in spec.legend():
        print(f"  {line}")

    print(f"\n  {'node':>5}  {'pitch':>9}  {'timbre':>9}  attributes")
    for tone in spec.tones:
        params = tone.params
        attrs = ", ".join(
            f"{k}={v:.3g}" for k, v in list(tone.attributes.items())[:3]
        )
        print(
            f"  {tone.node_id:>5}  {params.get('pitch', 0):>7.1f}Hz  "
            f"{params.get('timbre', '-'):>9}  {attrs}"
        )

    if result is not None:
        print(f"\n  in_node (Send pin):  {result['in_node']}")
        print(f"  out_node (Receive):  {result['out_node']}")
        resistances = np.asarray(result["resistances"], dtype=float)
        print(
            f"  resistors: {len(resistances)} "
            f"({resistances.min()/1e3:.0f}-{resistances.max()/1e3:.0f} kOhm)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sensing physicalization with matching sonification."
    )
    parser.add_argument(
        "--network",
        default="simple",
        choices=sorted(sample_networks.keys()),
        help="which sample network to build",
    )
    parser.add_argument(
        "--form",
        choices=["line", "bar", "scatter"],
        help="build a non-network form instead of a network",
    )
    parser.add_argument("--values", help="comma-separated values for --form line/bar")
    parser.add_argument("--label", default="value", help="name of the data attribute")
    parser.add_argument(
        "--primary", default="degree", help="attribute mapped to pitch"
    )
    parser.add_argument(
        "--secondary", default=None, help="attribute mapped to pulse rate"
    )
    parser.add_argument(
        "--pitch-set",
        default="pentatonic",
        help="pitch collection (pentatonic, major, minor, chromatic, whole_tone)",
    )
    parser.add_argument("--out", default="result", help="output directory")
    parser.add_argument(
        "--fast", action="store_true", help="fewer optimizer iterations, for iteration"
    )
    parser.add_argument(
        "--skip-stl", action="store_true", help="skip STL export (much faster)"
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="do not copy the spec into the calibration app",
    )
    args = parser.parse_args()

    if args.form:
        if args.form in ("line", "bar"):
            if not args.values:
                parser.error(f"--form {args.form} requires --values")
            values = [float(v) for v in args.values.split(",")]
            form = (
                line_graph_form(values, label=args.label)
                if args.form == "line"
                else bar_chart_form(values, label=args.label)
            )
        else:
            points = np.random.RandomState(0).rand(12, 2) * 100
            values = np.random.RandomState(1).rand(12) * 100
            form = scatter_form(points, values=values, label=args.label)

        spec = build_form(form, args.out, pitch_set=args.pitch_set)
        summarize(spec)
    else:
        network = sample_networks[args.network]
        nodes = list(network["nodes"])
        links = [list(l) for l in network["links"]]
        links.sort()

        result, spec = build_network(
            nodes,
            links,
            args.out,
            fast=args.fast,
            skip_stl=args.skip_stl,
            primary=args.primary,
            secondary=args.secondary,
            pitch_set=args.pitch_set,
        )
        summarize(spec, result)

    if not args.no_publish:
        published = publish_to_calibration(args.out)
        if published:
            print(f"\n  published to {published}")
            print("  run the calibration server, then open /explore to play it")

    print(f"\n  outputs in {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
