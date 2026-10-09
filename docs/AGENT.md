# Driving top-op from a Claude session

Everything the GUI does is reachable headless: through the MCP server (`topop mcp`, interactive) or
the CLI (`topop describe`, `topop run`, scripted). Both sit on `topop/agent.py` (`Session`) and share
`TOPOP_DATA_DIR` (default `~/.cache/topop`) with `topop serve`, so a run made here can be opened in the GUI.

## Setup

```
claude mcp add top-op -- uv run --directory /path/to/top-op topop mcp
```

Equivalent `.mcp.json` entry: `{"mcpServers": {"top-op": {"command": "uv", "args": ["run", "--directory", "/path/to/top-op", "topop", "mcp"]}}}`.
Optional `"env": {"TOPOP_DATA_DIR": "/some/dir"}` keeps projects somewhere else.

## The loop: describe, select, run, look

1. `load_mesh(path)` gives a `mesh_id`; `describe_mesh(mesh_id)` prints the bbox and a facet table
   (id, area, kind, normal or axis+radius, centroid, bbox of every face group, largest first). `preview_mesh` is the picture.
2. `create_project(name, design_mesh_id, elements_along_longest=30..40)`, then `add_support` and `add_load`
   with selections (below). Each call returns how many grid nodes the selection resolved to: 0 means it missed.
3. `voxel_stats(project_id)`: grid size, `n_active`, memory/time estimate, warnings, node counts per load/support.
4. `run(project_id, max_iter=5)` as a smoke test, then the real run. Compliance should fall and flatten, volume
   should sit at `volfrac`, `change` should drop below `tol` (0.01).
5. `result_preview(run_id)` (orange = result, grey = original). Try `view` `iso`, `+x`, `+y`, `+z`. Adjust
   (`set_params`, `set_grid`, `add_ref_model`, move a load), `run` again, then `export_stl(run_id, path)`.

CLI equivalent: `topop describe part.stl --png part.png`, write a case file, `topop run case.json --out out/`
(writes `result.stl`, `result.png`, `result.vti`, `density.npz`, `run.json`; `run.json` re-runs with `topop run`).
Exit codes: 0 done, 1 failed, 2 invalid case or project (issues printed), 3 out of memory. Overrides for the
design rules below: `--symmetry y` (or `y=12.5`, repeatable), `--overhang +z` (write `--overhang=-z` for negative
directions), `--stress-limit 200`, `--optimizer mma`; `--trim` clips `result.stl` to the CAD surface. The per-iteration
table gains `stress_max` (and `constraint` with a limit); the summary names the final max stress and, with a limit,
whether the constraint ended satisfied.
`export_case` / `load_case` convert between a project and a case file (`examples/*.json`: a `ProjectIn` whose
meshes are `path`s relative to the file; selections may use `"mesh_id": "design"`).

## Selections (loads, supports; coordinates in the mesh's own units)

| kind | JSON |
|---|---|
| facets | `{"kind":"facets","facet_ids":[0,3]}` ids from `describe_mesh`: planes and cylinders (add `"angle_deg"` if you changed it); STEP: B-rep faces |
| normal | `{"kind":"normal","direction":[0,0,1],"angle_deg":10,"within":[[x0,y0,z0],[x1,y1,z1]]}` |
| plane | `{"kind":"plane","point":[0,0,0],"normal":[1,0,0],"tol":0}` nodes on that plane, no mesh needed |
| box | `{"kind":"box","min":[..],"max":[..]}` or `{"kind":"box","center":[..],"size":[sx,sy,sz]}` |
| sphere | `{"kind":"sphere","center":[..],"radius":r}` |
| cylinder | `{"kind":"cylinder","center":[..],"radius":r,"height":h,"axis":"z"}` axis x, y, z or a vector |
| faces | `{"kind":"faces","face_ids":[..]}` raw triangle ids, GUI only |

`mesh_id` defaults to `"design"` (`"ref:<id>"` names a reference model). Primitives pick the surface grid nodes inside
them. Canonical primitive form (what is stored, what the GUI writes): `transform` = 16 floats **column-major**
(three.js `Matrix4.toArray()`, translation in elements 12..14) and `size`; box = unit cube scaled by `size`, sphere radius
= `size[0]`, cylinder axis = local **Y** with radius `size[0]`, height `size[1]`. Force is the total vector, split over the nodes.

Reference bodies (`add_ref_model`, any loaded mesh + `transform`): `keep_in` = forced solid (extends the domain),
`keep_out` = forced void (clearance).

## Holes: cylinder facets

Every facet row has a `kind`: `plane` (unit `normal`), `cylinder` (`axis` and `radius`, normal `[0,0,0]`: holes, bosses, round
fillets) or `other` (curved, normal 0). A hole is therefore the `cylinder` facet with the radius you want (radius is in mesh
units, so a Ø12 hole has `radius` 6; in `examples/bracket.stl` that is facet 8, and the four Ø6 holes have radius 3):
`{"kind":"facets","facet_ids":[8]}` selects its wall, e.g. as a bolt support or a pin load. Use `axis` and `centroid` to tell
holes of equal radius apart. `facet_faces(mesh_id, facet_id)` returns the triangle ids of a facet
(what the GUI highlights; the `faces` selection takes them, but `facets` does the same server-side).

## Design rules: symmetry, stress, overhang (`set_params`)

| field | effect |
|---|---|
| `optimizer` | `oc` (default: volume constraint only, fastest) or `mma` (any constraints, slower per iteration) |
| `symmetry` | mirror planes, `[{"axis":"y","position":null}]`; `position` is a world coordinate, `null` = centre of the design's bbox, snapped to the nearest voxel boundary or centre. Design variables are tied across the plane, so the result is exactly mirror-symmetric. Loads, supports and domain should be symmetric too (else `run` warns). `[]` removes |
| `stress_limit` | von Mises limit in the units of E (MCP: `0` removes). A p-norm aggregated constraint; forces `mma` (`voxel_stats` says so when `optimizer` is `oc`). `stress_pnorm` (default 64) is the FINAL exponent of an automatic p-continuation that starts at min(8, `stress_pnorm`) and doubles up to it as the run settles; the move limit is capped at 0.1/0.05 while the constraint is active. See docs/STRESS.md |
| `overhang` | additive-manufacturing build direction, `+x` .. `-z` (MCP: `"none"` removes): 45 degree Langelaar filter, every voxel must be supported from below. The base plate is the domain's **min** face along the axis for `+`, **max** face for `-`; no support structures are generated |

Reading the outcome: `run` returns `stress_max` (max von Mises of the last evaluated design) and `constraint` (`g`, null without a
limit: `g <= 0` is satisfied, the optimizer accepts up to 0.01; a larger value needs more iterations or a higher limit).
`result_stress_summary(run_id)` gives the final design's `max`, `mean_solid` (voxels with density >= 0.5) and the xyz `location`
of the max; the whole field is the `stress` array in `result.vti` and `density.npz` (and `GET /api/runs/{id}/stress`). These are
voxel stresses weighted by sqrt(density): sharp inner corners overshoot, so leave margin and compare relatively.

`export_stl(run_id, path, trim=true)` (CLI `--trim`) intersects the iso-surface with the design mesh: the part stays inside the
CAD surface and keeps its exact flat faces and hole walls where material reaches them. It needs a watertight design mesh; if that
or the boolean fails, the untrimmed STL is written and the reason is returned (`warnings`; REST: `X-Topop-Warnings`).

## STEP input (optional extra)

With `uv sync --extra step` (the OpenCascade kernel, `cadquery-ocp`), `load_mesh` and case-file `path`s also take `.step` / `.stp`
(mm as OpenCascade reports them, no rescaling). The file is tessellated once and cached; `describe_mesh` then lists
the **B-rep faces** as facets: `id` is the rank by exact area (descending), `brep_face` the face's index in the STEP file,
`kind` is `plane` / `cylinder` / `other`, and cylinders carry their exact `radius` and `axis`, so "the Ø12 hole" is the
`cylinder` facet with `radius` 6 and no guessing. `{"kind":"facets","facet_ids":[..]}` selects those faces exactly;
`angle_deg` is ignored, and ids are stable for a given file (they do not renumber like mesh facets). `normal`, `plane`
and the primitives work on the tessellation as for any mesh. Without the extra, uploading a STEP file fails with the install hint.

## Pitfalls

- **`within` boxes**: grid nodes sit up to h/2 outside the true surface and `within` clips strictly on node coordinates.
  Pad a box derived from the mesh bbox by one voxel `h` per side (`h` is in `voxel_stats`, about longest side / elements)
  and re-check after changing the resolution (h changes). Prefer `facets` or `plane` when a whole face will do.
- Facet ids are ranks by area for a given `angle_deg`; a changed angle renumbers them (STEP meshes: B-rep faces, never). Cylinders and other curved groups have normal 0: use `kind`, `radius` and `axis`.
- `overhang`: the base plate is the first grid layer that holds active cells along the build axis; cells of a keep-in
  body floating above a gap are unsupported by construction and will be removed.
- Always check the node counts before running; "resolves to zero nodes" blocks the run. Supports must stop all rigid motion.
- Size: keep `n_active` <= about 150k on a 16 GB machine (`est_bytes` and `est_sec_per_iter` in `voxel_stats`, about 30 KB per element).
  Debug at 30-40 elements along the longest side, refine to 60-100 at the end. Above the memory cap `run` fails fast (exit 3).
- Features thinner than about 2 voxels vanish; `rmin` (voxels, default 2) also limits the smallest member. Non-watertight
  meshes voxelize poorly (see `is_watertight`).
- Units are whatever the mesh is in; E, forces and lengths must be consistent. Compliance is only comparable between runs
  of the same setup, not across resolutions.
- `run` blocks until done and sends one progress notification per iteration; `cancel_run(run_id)` stops it and keeps the result.
