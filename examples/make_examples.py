"""Generate examples/cantilever.stl, bracket.stl and bracket.step (units: mm, min corner at origin).

bracket.step is the same part as bracket.stl plus a 3 mm fillet on the inner plate-wall edge; it
needs cadquery (`uv sync --extra examples`) and is skipped without it.

Run: uv run python examples/make_examples.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import trimesh

try:
    import cadquery as cq
except ImportError:  # optional extra
    cq = None

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


def make_bracket_step() -> cq.Workplane:
    plate = cq.Workplane("XY").box(80, 60, 10, centered=False)
    wall = cq.Workplane("XY").box(10, 60, 60, centered=False)
    # 3 mm blend on the inner plate-wall edge (x=10, z=10, along Y): the one curved non-hole face
    body = plate.union(wall).edges(cq.selectors.BoxSelector((9, -1, 9), (11, 61, 11))).fillet(3)
    cutters = [
        cq.Solid.makeCylinder(3, 30, cq.Vector(x, y, -10), cq.Vector(0, 0, 1))
        for x in (20, 72)
        for y in (8, 52)
    ]
    cutters.append(cq.Solid.makeCylinder(6, 30, cq.Vector(-10, 30, 35), cq.Vector(1, 0, 0)))
    return body.cut(cq.Compound.makeCompound(cutters))


def write_step(shape: cq.Workplane, name: str) -> None:
    solid = shape.val()
    assert solid.isValid(), f"{name} is not a valid solid"
    path = OUT / name
    cq.exporters.export(shape, str(path))
    # pin the header timestamp so regenerating the file does not change it
    text = re.sub(r"(FILE_NAME\('[^']*',')[^']*'", r"\g<1>2000-01-01T00:00:00'", path.read_text())
    path.write_text(text)
    print(f"{name}: {len(solid.Faces())} B-rep faces, volume {solid.Volume():.1f}")


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
    if cq is None:
        print("bracket.step skipped: cadquery is not installed (uv sync --extra examples)")
    else:
        write_step(make_bracket_step(), "bracket.step")
