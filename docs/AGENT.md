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
   (id, area, normal, centroid, bbox of every coplanar face group, largest first). `preview_mesh` is the picture.
2. `create_project(name, design_mesh_id, elements_along_longest=30..40)`, then `add_support` and `add_load`
   with selections (below). Each call returns how many grid nodes the selection resolved to: 0 means it missed.
3. `voxel_stats(project_id)`: grid size, `n_active`, memory/time estimate, warnings, node counts per load/support.
4. `run(project_id, max_iter=5)` as a smoke test, then the real run. Compliance should fall and flatten, volume
   should sit at `volfrac`, `change` should drop below `tol` (0.01).
5. `result_preview(run_id)` (orange = result, grey = original). Try `view` `iso`, `+x`, `+y`, `+z`. Adjust
   (`set_params`, `set_grid`, `add_ref_model`, move a load), `run` again, then `export_stl(run_id, path)`.

CLI equivalent: `topop describe part.stl --png part.png`, write a case file, `topop run case.json --out out/`
(writes `result.stl`, `result.png`, `result.vti`, `density.npz`, `run.json`; `run.json` re-runs with `topop run`).
Exit codes: 0 done, 1 failed, 2 invalid case or project (issues printed), 3 out of memory.
`export_case` / `load_case` convert between a project and a case file (`examples/*.json`: a `ProjectIn` whose
meshes are `path`s relative to the file; selections may use `"mesh_id": "design"`).

## Selections (loads, supports; coordinates in the mesh's own units)

| kind | JSON |
|---|---|
| facets | `{"kind":"facets","facet_ids":[0,3]}` ids from `describe_mesh` (add `"angle_deg"` if you changed it) |
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

## Pitfalls

- **`within` boxes**: grid nodes sit up to h/2 outside the true surface and `within` clips strictly on node coordinates.
  Pad a box derived from the mesh bbox by one voxel `h` per side (`h` is in `voxel_stats`, about longest side / elements)
  and re-check after changing the resolution (h changes). Prefer `facets` or `plane` when a whole face will do.
- Facet ids are ranks by area for a given `angle_deg`; a changed angle renumbers them. Curved groups have normal 0.
- Always check the node counts before running; "resolves to zero nodes" blocks the run. Supports must stop all rigid motion.
- Size: keep `n_active` <= about 150k on a 16 GB machine (`est_bytes` and `est_sec_per_iter` in `voxel_stats`, about 30 KB per element).
  Debug at 30-40 elements along the longest side, refine to 60-100 at the end. Above the memory cap `run` fails fast (exit 3).
- Features thinner than about 2 voxels vanish; `rmin` (voxels, default 2) also limits the smallest member. Non-watertight
  meshes voxelize poorly (see `is_watertight`).
- Units are whatever the mesh is in; E, forces and lengths must be consistent. Compliance is only comparable between runs
  of the same setup, not across resolutions.
- `run` blocks until done and sends one progress notification per iteration; `cancel_run(run_id)` stops it and keeps the result.
