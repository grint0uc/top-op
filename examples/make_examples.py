"""Generate examples/cantilever.stl and examples/bracket.stl (units: mm, min corner at origin).

Run: uv run python examples/make_examples.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).parent
SECTIONS = 64  # cylinder facets


def box(size: tuple[float, float, float], origin: tuple[float, float, float]) -> trimesh.Trimesh:
    """Axis-aligned box whose min corner is `origin`."""
    m = trimesh.creation.box(extents=size)
    m.apply_translation(np.asarray(origin) + np.asarray(size) / 2)
    return m


def cylinder(
    radius: float, height: float, center: tuple[float, float, float], axis: str
) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=SECTIONS)  # axis = Z
    if axis == "x":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == "y":
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    m.apply_translation(center)
    return m


def make_cantilever() -> trimesh.Trimesh:
    return box((60, 20, 20), (0, 0, 0))


def make_bracket() -> trimesh.Trimesh:
    plate = box((80, 60, 10), (0, 0, 0))
    wall = box((10, 60, 60), (0, 0, 0))  # rises 50 mm above the plate along the x=0 edge
    body = trimesh.boolean.union([plate, wall], engine="manifold")

    cutters = [
        # Ø6 through the plate (along Z), clear of the wall
        cylinder(3, 30, (x, y, 5), "z")
        for x in (20, 72)
        for y in (8, 52)
    ]
    # Ø12 through the wall (along X), centered on the wall face
    cutters.append(cylinder(6, 30, (5, 30, 35), "x"))
    return trimesh.boolean.difference([body, *cutters], engine="manifold")


def write(mesh: trimesh.Trimesh, name: str) -> None:
    assert mesh.is_watertight, f"{name} is not watertight"
    assert mesh.is_winding_consistent and mesh.volume > 0, f"{name} has bad winding"
    mesh.export(OUT / name)
    print(
        f"{name}: {len(mesh.faces)} faces, volume {mesh.volume:.1f}, bounds {mesh.bounds.tolist()}"
    )


if __name__ == "__main__":
    write(make_cantilever(), "cantilever.stl")
    write(make_bracket(), "bracket.stl")
