"""
Watertight CSG build of the three print parts (node / link / resistor).

Uses manifold3d, which guarantees closed, manifold output. Every invariant that
matters for a clean multi-material slice is checked here and raises RuntimeError
instead of silently writing a bad file:

  * each part is a valid, non-empty, closed solid with 0 open edges;
  * nothing is below the bed (z >= 0);
  * the link solid has the nodes, the resistor traces and the pins subtracted,
    so no volume is claimed by two materials;
  * the resistor traces are fully enclosed by link + node, except the pins.
"""

import numpy as np
import pyvista as pv

Z_HAT = np.array([0.0, 0.0, 1.0])


def _affine(origin, ax_x, ax_y, ax_z):
    """3x4 matrix for manifold3d: columns are the images of x, y, z, then offset."""
    return np.array(
        [
            [ax_x[0], ax_y[0], ax_z[0], origin[0]],
            [ax_x[1], ax_y[1], ax_z[1], origin[1]],
            [ax_x[2], ax_y[2], ax_z[2], origin[2]],
        ],
        dtype=float,
    )


def _box(m3, center, size_x, size_y, size_z, ax_x, ax_y, ax_z):
    return m3.Manifold.cube([size_x, size_y, size_z], True).transform(
        _affine(center, ax_x, ax_y, ax_z)
    )


def _cylinder(m3, p0, p1, radius, segments):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = np.linalg.norm(d)
    u = d / length
    ref = Z_HAT if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(u, ref)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    cyl = m3.Manifold.cylinder(length, radius, radius, segments, True)  # along +z
    return cyl.transform(_affine((p0 + p1) / 2, v, w, u))


def _unite(items):
    out = items[0]
    for it in items[1:]:
        out = out + it
    return out


def _to_polydata(man):
    mesh = man.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3].astype(float)
    tris = np.asarray(mesh.tri_verts)
    faces = np.hstack([np.full((len(tris), 1), 3), tris]).astype(np.int64).ravel()
    return pv.PolyData(verts, faces)


def build_print_parts(
    nodes,
    links,
    node_positions,
    node_radius,
    link_radius,
    node_center_z,
    link_axis_z,
    h_paths_all,
    v_boxes_all,
    resistor_links,
    pin_nodes,
    pin_radius=2.0,
    pin_length=6.0,
    h_width=0.5,
    v_width=1.0,
    sphere_segments=64,
    cyl_segments=48,
):
    """Returns ({'node','link','resistor'} -> pv.PolyData, report dict)."""
    import manifold3d as m3

    P = np.asarray(node_positions, float)

    def trim_bed(man):
        return man.trim_by_plane((0.0, 0.0, 1.0), 0.0)

    # nodes: spheres with a flat base
    node_solid = _unite(
        [
            trim_bed(
                m3.Manifold.sphere(node_radius, sphere_segments).translate(
                    (P[i][0], P[i][1], node_center_z)
                )
            )
            for i in nodes
        ]
    )

    # links: cylinders lying on the bed with a flat underside
    link_raw = _unite(
        [
            trim_bed(
                _cylinder(
                    m3,
                    (P[s][0], P[s][1], link_axis_z),
                    (P[t][0], P[t][1], link_axis_z),
                    link_radius,
                    cyl_segments,
                )
            )
            for s, t in links
        ]
    )

    # resistor traces; design-frame z (link axis = 0) -> print frame
    shift = np.array([0.0, 0.0, link_axis_z])
    trace_parts = []
    for h_paths in h_paths_all:
        for h_path in h_paths:
            pts = np.array(h_path, float)
            for i in range(1, len(pts)):
                seg = pts[i] - pts[i - 1]
                mag = np.linalg.norm(seg)
                if mag <= 0:
                    continue
                u = seg / mag
                side = np.cross(u, Z_HAT)
                if np.linalg.norm(side) < 1e-9:
                    side = np.array([1.0, 0.0, 0.0])
                side /= np.linalg.norm(side)
                up = np.cross(side, u)
                center = (pts[i] + pts[i - 1]) / 2 + shift
                trace_parts.append(
                    _box(m3, center, mag + h_width, h_width, h_width, u, side, up)
                )
    for v_boxes, (s, t) in zip(v_boxes_all, resistor_links):
        d = P[t] - P[s]
        d[2] = 0.0
        ax = d / np.linalg.norm(d)
        ay = np.cross(Z_HAT, ax)
        for vb in v_boxes:
            c = np.array(vb, float).mean(axis=0)
            z0, z1 = vb[0][2], vb[1][2]
            if abs(z1 - z0) <= 0:
                continue
            center = np.array([c[0], c[1], (z0 + z1) / 2]) + shift
            trace_parts.append(
                _box(m3, center, v_width, v_width, abs(z1 - z0), ax, ay, Z_HAT)
            )
    traces = _unite(trace_parts)

    # terminal pins: cylinder plus a web down to the bed
    pin_parts = []
    for n in pin_nodes:
        nbrs = [t if s == n else s for s, t in links if n in (s, t)]
        mean_vec = np.mean([P[m] - P[n] for m in nbrs], axis=0)
        mean_vec[2] = 0.0
        mean_vec /= np.linalg.norm(mean_vec)
        p0 = np.array([P[n][0], P[n][1], link_axis_z])
        p1 = p0 - (node_radius + pin_length) * mean_vec
        length = np.linalg.norm(p1 - p0)
        cyl = _cylinder(m3, p0, p1, pin_radius, cyl_segments)
        web = _box(
            m3,
            (p0 + p1) / 2 - np.array([0.0, 0.0, link_axis_z / 2]),
            length,
            2 * pin_radius,
            link_axis_z,
            -mean_vec,
            np.cross(mean_vec, Z_HAT),
            Z_HAT,
        )
        pin_parts.append(trim_bed(cyl + web))
    pins = _unite(pin_parts) if pin_parts else None

    resistor_solid = trim_bed(traces + pins if pins is not None else traces)

    # nothing is owned by two materials: carve conductors out of the link
    link_solid = link_raw - node_solid - resistor_solid

    # ---- invariants ----------------------------------------------------
    report = {}
    for name, man in (
        ("node", node_solid),
        ("link", link_solid),
        ("resistor", resistor_solid),
    ):
        if man.status() != m3.Error.NoError or man.is_empty():
            raise RuntimeError(f"{name}: invalid solid ({man.status()})")
        bb = man.bounding_box()  # (xmin, ymin, zmin, xmax, ymax, zmax)
        if bb[2] < -1e-6:
            raise RuntimeError(f"{name}: dips below the bed (zmin={bb[2]})")
        report[name] = dict(
            volume=float(man.volume()), zmin=float(bb[2]), zmax=float(bb[5])
        )

    overlap = (link_solid ^ (node_solid + resistor_solid)).volume()  # ^ is intersection
    if overlap > 1e-3:
        raise RuntimeError(f"link still overlaps conductive parts by {overlap:.4f} mm^3")
    report["link_conductor_overlap_mm3"] = float(overlap)

    outside = traces - link_raw - node_solid
    leak = (outside - pins).volume() if pins is not None else outside.volume()
    if leak > 1e-3:
        raise RuntimeError(f"resistor traces protrude from the links by {leak:.4f} mm^3")
    report["trace_leak_mm3"] = float(leak)

    parts = {
        "node": _to_polydata(node_solid),
        "link": _to_polydata(link_solid),
        "resistor": _to_polydata(resistor_solid),
    }
    for name, mesh in parts.items():
        if mesh.n_open_edges != 0:
            raise RuntimeError(f"{name}: {mesh.n_open_edges} open edges after export")
    return parts, report
