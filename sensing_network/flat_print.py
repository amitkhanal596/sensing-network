"""
Helpers that make a planar sensing network printable on a flat bed with
minimal support.

Print orientation: everything is built in a "design frame" where node centres
and link axes are at z = 0 (this is what the upstream pipeline produces). The
parts are then lifted by different amounts so that

  * nodes stay spheres, but get a flat base whose radius is R*sin(45 deg), so
    the whole lower hemisphere is a <= 45 degree overhang (no support);
  * links are lowered so they rest on the bed, with a flat underside of the
    same 45 degree construction (no bridging in mid air);
  * resistor traces ride at link-axis height, are re-centred inside the link
    cross-section, and the in/out pins get a web down to the bed.

All three meshes share one XY origin and the same bed plane (z = 0).
"""

import numpy as np
import pyvista as pv

from sensing_network.convert_utils import line_to_cylinder

OVERHANG_ANGLE = np.deg2rad(45.0)


def flat_cut_depth(radius, angle=OVERHANG_ANGLE):
    """Depth to slice off a circle/sphere so the flat has radius*sin(angle)."""
    return radius * (1.0 - np.cos(angle))


def clip_to_bed(mesh):
    """Remove everything below z = 0 and cap the cut so the mesh stays closed."""
    return mesh.clean().clip_closed_surface(normal="z", origin=(0, 0, 0)).clean()


def recentre_and_check_traces(
    h_paths_all,
    v_boxes_all,
    resistor_links,
    node_positions,
    node_radius,
    link_radius,
    axis_height,
    h_width,
    wall=0.3,
):
    """
    Shift each link's trace stack in z so it sits in the middle of the usable
    part of the (bed-clipped) link cross-section, then verify containment.

    Usable z range, in the design frame (link axis at 0):
        [-(axis_height) + wall + half_thickness, link_radius - wall - half_...]
    Returns (h_paths, v_boxes, report). Raises ValueError if a stack cannot fit.
    """
    lo_allowed = -axis_height + wall
    hi_allowed = link_radius - wall
    out_h, out_v, report = [], [], []

    for (s, t), h_paths, v_boxes in zip(resistor_links, h_paths_all, v_boxes_all):
        pts = [np.array(p) for hp in h_paths for p in hp]
        zmin = min(p[2] for p in pts) - h_width / 2
        zmax = max(p[2] for p in pts) + h_width / 2
        for b in v_boxes:
            zmin = min(zmin, b[0][2])
            zmax = max(zmax, b[1][2])
        span = zmax - zmin
        room = hi_allowed - lo_allowed
        if span > room:
            raise ValueError(
                f"link {s}-{t}: trace stack is {span:.2f} mm tall but only "
                f"{room:.2f} mm fits inside the link; increase link_radius or "
                f"vertical_step"
            )
        dz = (lo_allowed + hi_allowed) / 2 - (zmin + zmax) / 2
        out_h.append([[[p[0], p[1], p[2] + dz] for p in hp] for hp in h_paths])
        out_v.append([[[q[0], q[1], q[2] + dz] for q in b] for b in v_boxes])
        report.append(
            dict(link=(s, t), dz=float(dz), zmin=float(zmin + dz), zmax=float(zmax + dz))
        )

    # lateral + radial containment (outside the nodes)
    P = np.asarray(node_positions, dtype=float)
    problems = []
    for (s, t), h_paths in zip(resistor_links, out_h):
        a, b = P[s], P[t]
        ab = b - a
        for hp in h_paths:
            for p in hp:
                p = np.array(p)
                if min(np.linalg.norm(p - a), np.linalg.norm(p - b)) <= node_radius:
                    continue  # inside a node sphere -> intended contact
                u = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
                d = np.linalg.norm(p - (a + u * ab)) + h_width / 2
                if d > link_radius - 0.1:
                    problems.append((s, t, p.round(2).tolist(), round(float(d), 2)))
    if problems:
        raise ValueError(f"{len(problems)} trace points leave the link, e.g. {problems[:3]}")
    return out_h, out_v, report


def pin_with_web(p0, p1, pin_radius, axis_height, step=0.8):
    """Terminal pin = stack of overlapping cylinders from axis height to the bed,
    so the pin is supported by a solid web instead of hanging in air."""
    parts = []
    z = axis_height
    lowest = pin_radius  # centre height at which the pin bottom touches z = 0
    while z > lowest + 1e-9:
        parts.append(
            line_to_cylinder(
                np.array([p0[0], p0[1], z]), np.array([p1[0], p1[1], z]), pin_radius
            )
        )
        z -= step
    parts.append(
        line_to_cylinder(
            np.array([p0[0], p0[1], lowest]), np.array([p1[0], p1[1], lowest]), pin_radius
        )
    )
    return parts
