"""
Sonification for physicalization forms other than networks.

The sensing pipeline is built around a network: nodes are conductive touch
targets, links are the resistive paths between them. That structure is more
general than it looks, because many physicalizations reduce to "a set of
touchable regions connected in a chain or a mesh":

  * a line graph is a path graph -- one touch target per data point, each
    connected to the next;
  * a bar chart is a path graph over bars, ordered by category;
  * a scatterplot / spatial visualization is a set of points connected by a
    spanning structure chosen for fabrication rather than for meaning;
  * a choropleth is a region adjacency graph.

So the same fabrication pipeline can produce all of them, provided something
builds the right graph and keeps the mapping from graph node back to the
original data element. That is what this module does: each function returns
nodes, links, positions, and the per-element data, ready for both
`default_pipeline` and `default_sonification`.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

This file is part of the sonification extension to the sensing network
physicalization pipeline. The upstream pipeline is
Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae.
New work in this file: Copyright (c) 2026, the sonification extension authors.
Licensed under the same CC BY-NC-SA 4.0 terms as the upstream project.
"""

import numpy as np

from sensing_network.sonification.pipeline import default_sonification


def _path_links(n):
    return [[i, i + 1] for i in range(n - 1)]


def line_graph_form(values, x=None, label="value", z_scale=1.0, spacing=None):
    """
    A line graph as a chain of touchable points.

    The physical form follows the data: each point sits at its own height, so
    the shape can be traced by hand, while the sonification carries the exact
    value. Touch tells you the trend; sound tells you the number.

    Parameters
    ----------
    values : sequence of y-values.
    x : optional x positions (defaults to index).
    z_scale : vertical exaggeration applied to the physical height.
    spacing : horizontal distance between adjacent points. Defaults to a value
        that keeps the aspect ratio near 1, which prints and reads well.

    Returns
    -------
    dict with nodes, links, node_positions, attributes, and metadata.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError("a line graph needs at least two points")

    x = np.arange(n, dtype=float) if x is None else np.asarray(x, dtype=float)

    value_span = float(values.max() - values.min()) or 1.0
    if spacing is None:
        spacing = value_span * z_scale / max(n - 1, 1)

    x_pos = (x - x.min()) / (max(float(x.max() - x.min()), 1e-12)) * spacing * (n - 1)
    y_pos = (values - values.min()) * z_scale
    positions = np.column_stack([x_pos, y_pos, np.zeros(n)])

    return {
        "nodes": list(range(n)),
        "links": _path_links(n),
        "node_positions": positions,
        "attributes": {label: values, "x": x},
        "form": "line_graph",
        "metadata": {"label": label, "n_points": n},
    }


def bar_chart_form(values, categories=None, label="value", spacing=None):
    """
    A bar chart as a chain of touchable bar tops.

    Each bar's top is a touch target at the bar's height, so the profile can
    be felt directly and the value heard on contact.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 1:
        raise ValueError("a bar chart needs at least one bar")
    if categories is None:
        categories = [str(i) for i in range(n)]

    value_span = float(values.max() - values.min()) or float(abs(values.max())) or 1.0
    if spacing is None:
        spacing = value_span / max(n, 2)

    positions = np.column_stack(
        [
            np.arange(n, dtype=float) * spacing,
            values - values.min(),
            np.zeros(n),
        ]
    )

    return {
        "nodes": list(range(n)),
        "links": _path_links(n) if n > 1 else [],
        "node_positions": positions,
        "attributes": {label: values},
        "categories": list(categories),
        "form": "bar_chart",
        "metadata": {"label": label, "n_bars": n},
    }


def scatter_form(points, values=None, label="value", k=2, connect="mst"):
    """
    A spatial visualization: points in 2D or 3D, connected so the object holds
    together and the sensing chain can reach every point.

    `connect` chooses the structure:
      - "mst": minimum spanning tree, the least material that connects all
        points and the least visual clutter added to the data.
      - "knn": each point joined to its k nearest neighbours, which is sturdier
        but adds more links that a reader may mistake for data.

    The links here are structural, not semantic, which is worth stating in any
    caption: a reader tracing them is following fabrication, not relationships.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("points must be an (n, 2) or (n, 3) array")
    n = len(points)
    if n < 2:
        raise ValueError("a spatial form needs at least two points")

    if points.shape[1] == 2:
        positions = np.column_stack([points, np.zeros(n)])
    else:
        positions = points.copy()

    # Pairwise distances, used by both connection strategies.
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))

    links = []
    if connect == "knn":
        k = max(1, min(int(k), n - 1))
        seen = set()
        for i in range(n):
            order = np.argsort(dist[i])
            for j in order[1 : k + 1]:
                edge = (min(i, int(j)), max(i, int(j)))
                if edge not in seen:
                    seen.add(edge)
                    links.append([edge[0], edge[1]])
        # A k-NN graph can still be disconnected; the sensing chain requires
        # one component, so stitch any remaining pieces together.
        links = _ensure_connected(links, dist, n)
    else:
        links = _mst_links(dist, n)

    attributes = {}
    if values is not None:
        attributes[label] = np.asarray(values, dtype=float)

    return {
        "nodes": list(range(n)),
        "links": links,
        "node_positions": positions,
        "attributes": attributes,
        "form": "scatter",
        "metadata": {"label": label, "n_points": n, "connect": connect},
    }


def _mst_links(dist, n):
    """
    Prim's algorithm on a dense distance matrix.
    """
    in_tree = [0]
    remaining = set(range(1, n))
    links = []
    while remaining:
        best = None
        for i in in_tree:
            for j in remaining:
                d = dist[i, j]
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        links.append([min(i, j), max(i, j)])
        in_tree.append(j)
        remaining.discard(j)
    links.sort()
    return links


def _ensure_connected(links, dist, n):
    """
    Add the shortest links needed to make the graph a single component.
    """
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    for s, t in links:
        union(s, t)

    # Repeatedly join the two closest points in different components.
    while len({find(i) for i in range(n)}) > 1:
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j):
                    if best is None or dist[i, j] < best[0]:
                        best = (dist[i, j], i, j)
        _, i, j = best
        union(i, j)
        links.append([min(i, j), max(i, j)])

    links.sort()
    return links


def region_adjacency_form(adjacency, values=None, positions=None, label="value"):
    """
    A choropleth-style form: regions as touch targets, adjacency as links.

    `adjacency` is a dict of {region: [neighbouring regions]} or a list of
    (region_a, region_b) pairs. Region names are mapped to integer node IDs;
    the mapping is returned so values can be traced back.
    """
    if isinstance(adjacency, dict):
        names = list(adjacency.keys())
        for neighbours in adjacency.values():
            for nb in neighbours:
                if nb not in names:
                    names.append(nb)
        pairs = []
        for region, neighbours in adjacency.items():
            for nb in neighbours:
                pairs.append((region, nb))
    else:
        pairs = [tuple(p) for p in adjacency]
        names = []
        for a, b in pairs:
            if a not in names:
                names.append(a)
            if b not in names:
                names.append(b)

    index = {name: i for i, name in enumerate(names)}
    seen = set()
    links = []
    for a, b in pairs:
        i, j = index[a], index[b]
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        if edge not in seen:
            seen.add(edge)
            links.append([edge[0], edge[1]])
    links.sort()

    n = len(names)
    if positions is None:
        # Lay regions on a circle as a neutral default; a real map would pass
        # centroids instead.
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = np.column_stack(
            [np.cos(angles), np.sin(angles), np.zeros(n)]
        )
    else:
        positions = np.asarray(positions, dtype=float)
        if positions.shape[1] == 2:
            positions = np.column_stack([positions, np.zeros(len(positions))])

    attributes = {}
    if values is not None:
        if isinstance(values, dict):
            attributes[label] = np.array([float(values.get(nm, 0.0)) for nm in names])
        else:
            attributes[label] = np.asarray(values, dtype=float)

    return {
        "nodes": list(range(n)),
        "links": links,
        "node_positions": positions,
        "attributes": attributes,
        "region_names": names,
        "form": "region_adjacency",
        "metadata": {"label": label, "n_regions": n},
    }


def sonify_form(form, primary=None, **kwargs):
    """
    Build a sonification for any form produced by this module.

    The form's own data attribute is used as the primary channel by default,
    which is almost always what is wanted: for a line graph the point's value
    matters far more than its graph degree.
    """
    attributes = form.get("attributes", {})
    if primary is None:
        primary = form["metadata"].get("label")
        if primary not in attributes:
            primary = "degree"

    extra = {k: v for k, v in attributes.items() if k not in ("x",)}

    kwargs.setdefault("categorical", False)
    metadata = dict(form.get("metadata", {}))
    metadata["form"] = form.get("form")
    if "region_names" in form:
        metadata["region_names"] = form["region_names"]
    if "categories" in form:
        metadata["categories"] = form["categories"]

    return default_sonification(
        nodes=form["nodes"],
        links=form["links"],
        primary=primary,
        extra_attributes=extra if extra else None,
        metadata=metadata,
        **kwargs,
    )
