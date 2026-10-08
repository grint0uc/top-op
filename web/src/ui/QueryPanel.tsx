// The Query tab: build facets / normal / plane selections (the kinds an agent writes) from the GUI.
// The form lives in the store (queryForm); every edit re-derives selection.query, exactly like a primitive's fields.
import { useState } from 'react';
import type { FacetInfo } from '../api/client';
import { fetchFacets, hoverFacet, setNormalWithin, setQueryKind, toggleFacet } from '../state/actions';
import { designEntry, facetKey } from '../state/derived';
import { AXIS_PRESETS, type QueryForm, type QueryKind, type Vec3, dirName, formToSelection, queryLabel } from '../state/query';
import { useStore } from '../state/store';
import { Btn, Field, NumField } from './controls';

const KINDS: { id: QueryKind; label: string; hint: string }[] = [
  { id: 'normal', label: 'Normal', hint: 'All surface faces pointing along a direction (optionally inside a box)' },
  { id: 'plane', label: 'Plane', hint: 'Grid nodes near a plane; no mesh needed' },
  { id: 'facets', label: 'Facets', hint: 'Coplanar face groups from the facet table, largest first' },
];

const AXES = ['x', 'y', 'z'] as const;
const TOP_FACETS = 12;

const edit = (fn: (f: QueryForm) => QueryForm) => useStore.getState().editQueryForm(fn);

/** Three number fields editing one component each of a vector. */
function VecFields({ value, onChange, testId }: { value: readonly number[]; onChange: (v: Vec3) => void; testId: string }) {
  return (
    <div className="vec3">
      {AXES.map((a, k) => (
        <NumField
          key={a}
          value={value[k] ?? 0}
          testId={`${testId}-${a}`}
          title={a}
          onChange={(v) => onChange(value.map((x, i) => (i === k ? v : x)) as Vec3)}
        />
      ))}
    </div>
  );
}

function DirPresets({ value, onPick, testId }: { value: readonly number[]; onPick: (v: Vec3) => void; testId: string }) {
  const current = dirName(value);
  return (
    <div className="toolbar-row">
      {AXIS_PRESETS.map((p) => (
        <Btn key={p.label} active={current === p.label} onClick={() => onPick([...p.v])} testId={`${testId}-${p.label}`}>
          {p.label}
        </Btn>
      ))}
    </div>
  );
}

function NormalForm({ form }: { form: QueryForm['normal'] }) {
  const h = useStore((s) => s.voxel.stats?.h);
  return (
    <div className="query-form" data-testid="query-normal">
      <span className="dim">Direction</span>
      <DirPresets value={form.dir} onPick={(dir) => edit((f) => ({ ...f, normal: { ...f.normal, dir } }))} testId="normal-dir" />
      <VecFields value={form.dir} onChange={(dir) => edit((f) => ({ ...f, normal: { ...f.normal, dir } }))} testId="normal-vec" />
      <Field label="Angle (degrees from the direction)">
        <NumField
          value={form.angle}
          min={0.1}
          max={180}
          step={1}
          testId="normal-angle"
          onChange={(angle) => edit((f) => ({ ...f, normal: { ...f.normal, angle } }))}
        />
      </Field>
      <label className="check">
        <input type="checkbox" checked={form.within !== null} data-testid="normal-within-toggle" onChange={(e) => setNormalWithin(e.target.checked)} />
        Only inside a box (within)
      </label>
      {form.within && (
        <div className="query-box" data-testid="normal-within">
          <span className="dim">min</span>
          <VecFields
            value={form.within.min}
            testId="within-min"
            onChange={(min) => edit((f) => ({ ...f, normal: { ...f.normal, within: f.normal.within && { ...f.normal.within, min } } }))}
          />
          <span className="dim">max</span>
          <VecFields
            value={form.within.max}
            testId="within-max"
            onChange={(max) => edit((f) => ({ ...f, normal: { ...f.normal, within: f.normal.within && { ...f.normal.within, max } } }))}
          />
          <p className="dim" data-testid="within-hint">
            The box clips grid <em>nodes</em>, and nodes sit up to h/2 outside the surface, so a box taken from the mesh
            bbox must be padded: it was pre-filled with the design bbox grown by one voxel{h ? ` (h = ${h.toPrecision(4)})` : ''}. Shrink
            it to the region you want, keeping that margin on the sides that touch the surface.
          </p>
        </div>
      )}
    </div>
  );
}

function PlaneForm({ form }: { form: QueryForm['plane'] }) {
  return (
    <div className="query-form" data-testid="query-plane">
      <span className="dim">Point on the plane</span>
      <VecFields value={form.point} testId="plane-point" onChange={(point) => edit((f) => ({ ...f, plane: { ...f.plane, point } }))} />
      <span className="dim">Normal</span>
      <DirPresets value={form.normal} onPick={(normal) => edit((f) => ({ ...f, plane: { ...f.plane, normal } }))} testId="plane-dir" />
      <VecFields value={form.normal} testId="plane-normal" onChange={(normal) => edit((f) => ({ ...f, plane: { ...f.plane, normal } }))} />
      <Field label="Tolerance (0 = the single node layer nearest the plane, +-h/2)">
        <NumField value={form.tol} min={0} step={0.5} testId="plane-tol" onChange={(tol) => edit((f) => ({ ...f, plane: { ...f.plane, tol } }))} />
      </Field>
    </div>
  );
}

const fmt = (v: readonly number[]): string => v.map((x) => +x.toPrecision(3)).join(', ');

function FacetRow({ facet, picked }: { facet: FacetInfo; picked: boolean }) {
  return (
    <li
      className={`facet-row ${picked ? 'picked' : ''}`}
      data-testid="facet-row"
      data-facet-id={facet.id}
      onMouseEnter={() => hoverFacet(facet)}
      onMouseLeave={() => hoverFacet(null)}
      onClick={() => toggleFacet(facet.id)}
    >
      <input type="checkbox" checked={picked} readOnly tabIndex={-1} aria-label={`facet ${facet.id}`} />
      <span className="facet-id">#{facet.id}</span>
      <span className="facet-area" title="area">
        {facet.area.toPrecision(4)}
      </span>
      <span className={`facet-kind kind-${facet.kind}`} data-testid="facet-kind" data-kind={facet.kind} title="surface kind">
        {facet.kind}
      </span>
      {facet.kind === 'cylinder' ? (
        <span title={facet.axis ? `axis ${dirName(facet.axis)}` : 'radius'}>
          <span data-testid="facet-radius">r {+(facet.radius ?? 0).toPrecision(4)}</span>
          {facet.axis ? ` ax ${dirName(facet.axis)}` : ''}
        </span>
      ) : facet.kind === 'plane' ? (
        <span title="normal">n {dirName(facet.normal)}</span>
      ) : (
        <span />
      )}
      <span className="dim" title="centroid">
        c ({fmt(facet.centroid)})
      </span>
      <span className="dim" title="triangles">
        {facet.n_faces} tri
        {facet.brep_face != null && (
          <span data-testid="facet-brep" title="B-rep face index in the STEP file">
            {' '}
            &middot; B-rep #{facet.brep_face}
          </span>
        )}
      </span>
    </li>
  );
}

function FacetsForm({ form }: { form: QueryForm['facets'] }) {
  const meshId = useStore((s) => s.project.design_mesh?.mesh_id ?? null);
  const table = useStore((s) => (meshId ? s.facetCache[facetKey(meshId, form.angle)] : undefined));
  const ui = useStore((s) => s.queryUi);
  const step = useStore((s) => (meshId ? s.meshes[meshId]?.info.source === 'step' : false));
  const nBrep = useStore((s) => (meshId ? s.meshes[meshId]?.info.n_brep_faces : null));
  const [all, setAll] = useState(false);
  const rows = table ? (all ? table.facets : table.facets.slice(0, TOP_FACETS)) : [];
  return (
    <div className="query-form" data-testid="query-facets">
      {step && (
        <p className="badge-row" data-testid="facets-source">
          <span className="badge badge-step" title="The mesh came from a STEP file: facets are its exact B-rep faces, whatever the angle">
            STEP: ids are B-rep faces
          </span>
          {nBrep != null && <span className="dim">{nBrep} faces in the file</span>}
        </p>
      )}
      <div className="toolbar-row">
        <Field label={step ? 'Coplanarity angle (ignored for STEP)' : 'Coplanarity angle (degrees)'}>
          <NumField
            value={form.angle}
            min={0}
            max={90}
            step={1}
            disabled={step}
            testId="facets-angle"
            onChange={(angle) => edit((f) => ({ ...f, facets: { angle, ids: [] } }))}
          />
        </Field>
        <Btn onClick={() => void fetchFacets()} disabled={!meshId || ui.loading} testId="facets-fetch" title="GET /api/meshes/{id}/facets">
          {ui.loading ? 'Fetching...' : table ? 'Refresh' : 'Fetch facets'}
        </Btn>
      </div>
      {ui.error && (
        <p className="red" data-testid="facets-error">
          {ui.error}
        </p>
      )}
      {table && (
        <p className="dim" data-testid="facets-summary">
          {table.n_facets_total} facets at {table.angle_deg}&deg;{table.n_facets_total > table.facets.length ? ` (largest ${table.facets.length} listed)` : ''}. Hover a row to
          see its faces, click to select.
        </p>
      )}
      {!table && !ui.loading && <p className="dim">Fetch the facet table to list the largest planar regions.</p>}
      {rows.length > 0 && (
        <ul className="facet-list" data-testid="facet-list">
          {rows.map((f) => (
            <FacetRow key={f.id} facet={f} picked={form.ids.includes(f.id)} />
          ))}
        </ul>
      )}
      {table && table.facets.length > TOP_FACETS && (
        <Btn onClick={() => setAll(!all)} testId="facets-more">
          {all ? `Top ${TOP_FACETS} only` : `Show all ${table.facets.length}`}
        </Btn>
      )}
    </div>
  );
}

export function QueryPanel() {
  const form = useStore((s) => s.queryForm);
  const meshId = useStore((s) => s.project.design_mesh?.mesh_id ?? null);
  const query = useStore((s) => s.selection.query);
  const hasDesign = useStore((s) => designEntry(s) !== null);
  const valid = formToSelection(form, meshId) !== null;
  const { commitQueryForm } = useStore.getState();

  return (
    <div className="query-panel" data-testid="query-panel">
      <div className="toolbar-row">
        {KINDS.map((k) => (
          <Btn key={k.id} active={form.kind === k.id} onClick={() => setQueryKind(k.id)} testId={`query-tab-${k.id}`} title={k.hint} disabled={!hasDesign}>
            {k.label}
          </Btn>
        ))}
      </div>
      {form.kind === 'normal' && <NormalForm form={form.normal} />}
      {form.kind === 'plane' && <PlaneForm form={form.plane} />}
      {form.kind === 'facets' && <FacetsForm form={form.facets} />}
      <div className="toolbar-row">
        <Btn onClick={commitQueryForm} disabled={!valid} testId="query-apply" title="Make this query the current selection (it already follows edits)">
          Select
        </Btn>
        <span className={valid ? 'dim' : 'amber'} data-testid="query-status">
          {query ? queryLabel(query) : valid ? 'not selected yet' : form.kind === 'facets' ? 'pick at least one facet' : 'incomplete'}
        </span>
      </div>
    </div>
  );
}
