"""
Node attributes available for sonification.

A physicalization already encodes structure in its geometry: a listener can
feel a node's position and trace its links. What touch conveys poorly is
anything *derived* -- centrality, community membership, a value attached to a
node from an external table. Those are what sound is best spent on, so this
module computes them and describes them in words for a verbal readout.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import numpy as np
import networkx as nx

from sensing_network.convert_utils import to_nxgraph


def _safe(fn, G, nodes, default=0.0):
    """
    Run a networkx metric, falling back to a constant if it fails to converge
    (e.g. eigenvector centrality on a disconnected or bipartite graph).
    """
    try:
        result = fn(G)
    except Exception:
        return np.full(len(nodes), float(default))
    return np.array([float(result.get(n, default)) for n in nodes])


def _degree(G, nodes):
    return np.array([float(G.degree(n)) for n in nodes])


def _weighted_degree(G, nodes):
    if len(nx.get_edge_attributes(G, "weight")) == 0:
        return _degree(G, nodes)
    return np.array([float(G.degree(n, weight="weight")) for n in nodes])


def _betweenness(G, nodes):
    return _safe(lambda g: nx.betweenness_centrality(g), G, nodes)


def _closeness(G, nodes):
    return _safe(lambda g: nx.closeness_centrality(g), G, nodes)


def _eigenvector(G, nodes):
    return _safe(lambda g: nx.eigenvector_centrality_numpy(g), G, nodes)


def _pagerank(G, nodes):
    return _safe(lambda g: nx.pagerank(g), G, nodes)


def _clustering(G, nodes):
    return _safe(lambda g: nx.clustering(g), G, nodes)


def _core_number(G, nodes):
    def core(g):
        g = nx.Graph(g)
        g.remove_edges_from(nx.selfloop_edges(g))
        return nx.core_number(g)

    return _safe(core, G, nodes)


def _community(G, nodes):
    """
    Greedy modularity communities, returned as integer labels.
    """
    try:
        communities = nx.community.greedy_modularity_communities(G)
    except Exception:
        return np.zeros(len(nodes))
    label = {}
    for i, group in enumerate(communities):
        for n in group:
            label[n] = i
    return np.array([float(label.get(n, 0)) for n in nodes])


def _component(G, nodes):
    label = {}
    for i, group in enumerate(nx.connected_components(G)):
        for n in group:
            label[n] = i
    return np.array([float(label.get(n, 0)) for n in nodes])


# Each entry: (function, human-readable name, whether values are categorical,
# and a short phrase describing what a high value means).
ATTRIBUTE_FUNCTIONS = {
    "degree": (_degree, "degree", False, "many direct connections"),
    "weighted_degree": (
        _weighted_degree,
        "weighted degree",
        False,
        "strong total connection weight",
    ),
    "betweenness": (
        _betweenness,
        "betweenness centrality",
        False,
        "lies on many shortest paths",
    ),
    "closeness": (
        _closeness,
        "closeness centrality",
        False,
        "close to all other nodes",
    ),
    "eigenvector": (
        _eigenvector,
        "eigenvector centrality",
        False,
        "connected to other well-connected nodes",
    ),
    "pagerank": (_pagerank, "PageRank", False, "high random-walk importance"),
    "clustering": (
        _clustering,
        "clustering coefficient",
        False,
        "neighbours are interconnected",
    ),
    "core_number": (_core_number, "k-core number", False, "sits in a dense core"),
    "community": (_community, "community", True, "community membership"),
    "component": (_component, "connected component", True, "component membership"),
}


# Preferred mapping direction per attribute, following Walker (2002): the
# polarity listeners expect depends on the data dimension, not on the audio
# channel. All the graph metrics here are "more is more prominent", so they
# take positive polarity. Attributes where listeners tend to expect the
# opposite (notably *size*, which Walker found maps to *falling* pitch) should
# be passed with polarity=-1 by the caller.
#
# This table is a default, not a finding: the right polarity for a given
# dataset is an empirical question, which is why `default_sonification` exposes
# a `polarity` argument.
ATTRIBUTE_POLARITY = {name: 1 for name in [
    "degree",
    "weighted_degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "clustering",
    "core_number",
]}

# Attribute names that commonly carry a negative polarity preference. Matched
# as substrings against user-supplied attribute names, and only used to *warn*,
# never to silently flip a mapping.
NEGATIVE_POLARITY_HINTS = ("size", "radius", "diameter", "width", "area", "length")


def suggest_polarity(name):
    """
    The polarity this attribute probably wants, or None when unknown.
    """
    if name in ATTRIBUTE_POLARITY:
        return ATTRIBUTE_POLARITY[name]
    lowered = name.lower()
    if any(hint in lowered for hint in NEGATIVE_POLARITY_HINTS):
        return -1
    return None


def describe_attribute(name):
    """
    Return (label, is_categorical, meaning) for an attribute name.
    """
    if name in ATTRIBUTE_FUNCTIONS:
        _, label, categorical, meaning = ATTRIBUTE_FUNCTIONS[name]
        return label, categorical, meaning
    # An attribute supplied by the user: its name is the best label available,
    # and "higher" is all that can honestly be said about its direction.
    readable = name.replace("_", " ")
    return readable, False, f"a higher {readable}"


def node_attributes(nodes, links, attributes=None, extra=None, link_weights=None):
    """
    Compute per-node attributes.

    Parameters
    ----------
    nodes, links : as used throughout the pipeline.
    attributes : list of names from ATTRIBUTE_FUNCTIONS. Defaults to a
        general-purpose set.
    extra : dict of {name: sequence} for data brought in from outside the
        graph (e.g. a measured value per node). Values are aligned to `nodes`
        by position.
    link_weights : optional per-link weights, used by weighted attributes.

    Returns
    -------
    dict of {attribute_name: np.ndarray aligned to `nodes`}
    """
    if attributes is None:
        attributes = ["degree", "betweenness", "clustering", "community"]

    G = to_nxgraph(nodes, links, link_weights)
    node_list = list(nodes)

    result = {}
    for name in attributes:
        if name not in ATTRIBUTE_FUNCTIONS:
            raise ValueError(
                f"unknown attribute '{name}'. "
                f"Available: {sorted(ATTRIBUTE_FUNCTIONS)}"
            )
        fn = ATTRIBUTE_FUNCTIONS[name][0]
        result[name] = fn(G, node_list)

    if extra:
        for name, values in extra.items():
            values = np.asarray(values, dtype=float)
            if len(values) != len(node_list):
                raise ValueError(
                    f"extra attribute '{name}' has {len(values)} values "
                    f"but there are {len(node_list)} nodes"
                )
            result[name] = values

    return result
