from __future__ import annotations

import base64
import dataclasses
import io
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh
from PIL import Image

from topop.core.benchmarks import cantilever, cantilever_params
from topop.core.export import (
    VIEWS,
    density_to_mesh,
    from_npz_bytes,
    render_png,
    to_npz_bytes,
    to_stl_bytes,
    to_vti_bytes,
    trim_to_design,
)
from topop.core.optimize import optimize
from topop.core.problem import Grid
from topop.core.voxelize import build_domain, load_mesh

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def box_case():
    grid = Grid(origin=(1.0, -2.0, 0.5), h=0.5, shape=(10, 8, 6))
    rho = np.zeros(grid.shape)
    rho[2:7, 1:5, 1:4] = 1.0
    lo = np.asarray(grid.origin) + grid.h * np.array([2, 1, 1])
    hi = np.asarray(grid.origin) + grid.h * np.array([7, 5, 4])
    return grid, rho, np.stack([lo, hi])


@pytest.fixture(scope="module")
def bracket(examples_dir):
    return load_mesh(examples_dir / "bracket.stl")


def test_density_box_bounds(box_case):
    grid, rho, bounds = box_case
    mesh = density_to_mesh(rho, grid)
    assert np.allclose(mesh.bounds, bounds, atol=1e-9)
    assert mesh.is_watertight and mesh.volume > 0
    # marching cubes chamfers edges and corners, so the volume is a bit below the box volume
    assert 0.8 * np.prod(bounds[1] - bounds[0]) < mesh.volume <= np.prod(bounds[1] - bounds[0])

    smooth = density_to_mesh(rho, grid, smooth_iters=3)
    assert len(smooth.faces) == len(mesh.faces) and smooth.is_watertight
    assert np.all(smooth.bounds[0] >= bounds[0] - grid.h) and np.all(
        smooth.bounds[1] <= bounds[1] + grid.h
    )


def test_density_threshold_and_empty(box_case):
    grid, rho, _ = box_case
    assert len(density_to_mesh(np.zeros(grid.shape), grid).faces) == 0
    assert len(density_to_mesh(0.4 * rho, grid, threshold=0.5).faces) == 0
    assert len(density_to_mesh(0.4 * rho, grid, threshold=0.2).faces) > 0
    with pytest.raises(ValueError):
        density_to_mesh(np.zeros((2, 2, 2)), grid)
    with pytest.raises(ValueError):
        density_to_mesh(rho, grid, threshold=0.0)


def test_stl_bytes_round_trip(box_case):
    grid, rho, bounds = box_case
    mesh = density_to_mesh(rho, grid)
    data = to_stl_bytes(mesh)
    assert len(data) == 84 + 50 * len(mesh.faces)  # binary STL
    back = trimesh.load(io.BytesIO(data), file_type="stl")
    assert back.is_watertight and len(back.faces) == len(mesh.faces)
    assert np.allclose(back.bounds, bounds, atol=1e-5)
    assert len(to_stl_bytes(trimesh.Trimesh())) == 84


@pytest.fixture(scope="module")
def cantilever_result():
    problem = cantilever(20, 8, 4)
    res = optimize(problem, dataclasses.replace(cantilever_params(), max_iter=15))
    design = trimesh.creation.box(extents=(20, 8, 4))
    design.apply_translation((10, 4, 2))
    return res.rho, problem.grid, design


def test_trim_to_design_cantilever(cantilever_result):
    rho, grid, design = cantilever_result
    # threshold < 0.5 puts the iso-surface outside the outer element faces where rho ~ 1
    raw = density_to_mesh(rho, grid, threshold=0.3)
    assert np.any(raw.bounds[0] < -0.05) and np.any(raw.bounds[1] > design.bounds[1] + 0.05)
    out, warnings = trim_to_design(raw, design)
    assert warnings == []
    assert out.is_watertight and out.volume > 0
    assert np.all(out.bounds[0] >= design.bounds[0] - 1e-6)
    assert np.all(out.bounds[1] <= design.bounds[1] + 1e-6)
    assert out.volume <= raw.volume
    # the CAD skin is kept exactly where the part touches it (e.g. the clamped face x=0)
    assert np.isclose(out.bounds[0], 0, atol=1e-9).all()
    # an inside-out (but closed) design trims the same
    flipped = design.copy()
    flipped.invert()
    out2, warnings2 = trim_to_design(raw, flipped)
    assert warnings2 == [] and out2.volume == pytest.approx(out.volume, rel=1e-9)


def test_trim_to_design_failures(cantilever_result):
    rho, grid, design = cantilever_result
    raw = density_to_mesh(rho, grid)
    open_design = trimesh.Trimesh(design.vertices, design.faces[:-2], process=False)
    assert not open_design.is_watertight
    out, warnings = trim_to_design(raw, open_design)
    assert out is raw and len(warnings) == 1 and "design" in warnings[0]
    open_result = trimesh.Trimesh(raw.vertices, raw.faces[:-3], process=False)
    out, warnings = trim_to_design(open_result, design)
    assert out is open_result and len(warnings) == 1 and "result" in warnings[0]
    far = design.copy()
    far.apply_translation((100, 0, 0))
    out, warnings = trim_to_design(raw, far)
    assert out is raw and len(warnings) == 1 and "empty" in warnings[0]
    empty = trimesh.Trimesh()
    out, warnings = trim_to_design(empty, design)
    assert out is empty and len(warnings) == 1


def test_trim_to_design_bracket_timing(bracket):
    grid, active, _, _ = build_domain(bracket, [], 180)
    raw = density_to_mesh(active.astype(float), grid)
    assert len(raw.faces) >= 200_000
    t = time.perf_counter()
    out, warnings = trim_to_design(raw, bracket)
    assert time.perf_counter() - t < 5.0
    assert warnings == [] and out.is_watertight
    assert np.all(out.bounds[0] >= bracket.bounds[0] - 1e-6)
    assert np.all(out.bounds[1] <= bracket.bounds[1] + 1e-6)
    assert out.volume <= min(raw.volume, bracket.volume) + 1e-6 * bracket.volume
    assert out.volume > 0.95 * bracket.volume


def _decode_vtk(text: str, dtype) -> np.ndarray:
    raw = base64.b64decode("".join(text.split()))
    nbytes = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    assert len(raw) == 8 + nbytes
    return np.frombuffer(raw[8:], dtype=dtype)


def test_vti(box_case):
    grid, rho, _ = box_case
    rho = rho * np.linspace(0.1, 1.0, grid.nel).reshape(grid.shape)
    passive = np.zeros(grid.shape, dtype=np.int8)
    passive[0, 0, 0], passive[-1, -1, -1] = 1, -1
    root = ET.fromstring(to_vti_bytes(rho, passive, grid))
    assert root.tag == "VTKFile" and root.attrib["type"] == "ImageData"
    assert root.attrib["header_type"] == "UInt64" and root.attrib["byte_order"] == "LittleEndian"
    image = root.find("ImageData")
    assert image.attrib["WholeExtent"] == "0 10 0 8 0 6"
    assert np.allclose([float(v) for v in image.attrib["Origin"].split()], grid.origin)
    assert np.allclose([float(v) for v in image.attrib["Spacing"].split()], grid.h)
    arrays = {a.attrib["Name"]: a for a in image.iter("DataArray")}
    assert arrays["density"].attrib["type"] == "Float32"
    assert arrays["passive"].attrib["type"] == "Int8"
    dens = _decode_vtk(arrays["density"].text, "<f4")
    pas = _decode_vtk(arrays["passive"].text, np.int8)
    assert dens.size == pas.size == grid.nx * grid.ny * grid.nz
    # VTK cell order: x fastest
    assert np.allclose(dens.reshape(grid.shape, order="F"), rho, atol=1e-7)
    assert np.array_equal(pas.reshape(grid.shape, order="F"), passive)
    assert pas[0] == 1 and pas[-1] == -1


def test_npz_round_trip(box_case):
    grid, rho, _ = box_case
    active = rho > 0
    passive = np.zeros(grid.shape, dtype=np.int8)
    passive[3, 2, 2] = 1
    rho2, grid2, active2, passive2 = from_npz_bytes(to_npz_bytes(rho, grid, active, passive))
    assert grid2 == grid
    assert np.array_equal(rho2, rho) and np.array_equal(active2, active)
    assert np.array_equal(passive2, passive) and passive2.dtype == np.int8


def _pixels(png: bytes) -> np.ndarray:
    assert png.startswith(PNG_MAGIC)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


def test_render_png_bracket(bracket):
    t = time.perf_counter()
    png = render_png([(bracket, (0.6, 0.7, 0.9), 1.0)])
    assert time.perf_counter() - t < 3.0
    img = _pixels(png)
    assert img.shape == (700, 900, 3)
    colours = np.unique(img.reshape(-1, 3), axis=0)
    assert len(colours) > 10
    # the model covers a good part of the frame
    bg = np.all(img == img[5, -5], axis=2)
    assert 0.15 < 1 - bg.mean() < 0.9


def test_render_png_views_layers_and_bounds(bracket, examples_dir):
    grid, active, _, _ = build_domain(bracket, [], 40)
    result = density_to_mesh(active.astype(float), grid)
    keep = trimesh.creation.box(extents=(20, 20, 20))
    keep.apply_translation((60, 30, 30))
    layers = [
        (bracket, (0.7, 0.7, 0.7), 0.25),
        (result, (0.9, 0.4, 0.2), 1.0),
        (keep, (0, 1, 0), 0.4),
    ]
    for view in VIEWS:
        img = _pixels(render_png(layers, view=view, size=(320, 240)))
        assert img.shape == (240, 320, 3)
        assert len(np.unique(img.reshape(-1, 3), axis=0)) > 5
    # fixed bounds keep the framing; the empty scene still renders
    wide = _pixels(
        render_png([(bracket, (1, 0, 0), 1.0)], bounds=[[-100, -100, -100], [200, 200, 200]])
    )
    tight = _pixels(render_png([(bracket, (1, 0, 0), 1.0)]))
    assert (wide[..., 0] > 200).sum() > 0 and (np.all(wide == wide[5, -5], axis=2)).mean() > (
        np.all(tight == tight[5, -5], axis=2)
    ).mean()
    assert _pixels(render_png([], view="+z")).shape == (700, 900, 3)
    with pytest.raises(ValueError):
        render_png([(bracket, (1, 0, 0), 1.0)], view="sideways")
