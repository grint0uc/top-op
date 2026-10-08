import type { GridSpec, MaterialSpec, ParamsSpec } from '../api/client';
import type { ProjectDoc } from './store';

export const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export const DEFAULT_GRID: GridSpec = { elements_along_longest: 60, padding: 1 };
export const DEFAULT_MATERIAL: MaterialSpec = { E: 1, nu: 0.3 };
export const DEFAULT_PARAMS: ParamsSpec = {
  volfrac: 0.3,
  penal: 3,
  rmin: 2,
  max_iter: 100,
  tol: 0.01,
  move: 0.2,
  heaviside: false,
  continuation: false,
  solver: 'auto',
  dtype: 'float64',
  density_every: 1,
};

export function defaultProject(): ProjectDoc {
  return {
    name: 'untitled',
    design_mesh: null,
    ref_models: [],
    grid: { ...DEFAULT_GRID },
    material: { ...DEFAULT_MATERIAL },
    params: { ...DEFAULT_PARAMS },
    loads: [],
    supports: [],
  };
}
