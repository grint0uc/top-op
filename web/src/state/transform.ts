// Column-major 4x4 <-> translation / Euler (XYZ, degrees) / scale, shared by the primitive and reference-model fields.
import { Euler, Matrix4, Quaternion, Vector3 } from 'three';

const RAD = Math.PI / 180;

export interface TRS {
  pos: number[];
  rot: number[]; // degrees, Euler order XYZ
  scale: number[];
}

export function decomposeTRS(m: readonly number[]): TRS {
  const p = new Vector3();
  const q = new Quaternion();
  const s = new Vector3();
  new Matrix4().fromArray(m as number[]).decompose(p, q, s);
  const e = new Euler().setFromQuaternion(q, 'XYZ');
  return { pos: [p.x, p.y, p.z], rot: [e.x / RAD, e.y / RAD, e.z / RAD], scale: [s.x, s.y, s.z] };
}

export function composeTRS(pos: readonly number[], rot: readonly number[], scale: readonly number[] = [1, 1, 1]): number[] {
  const q = new Quaternion().setFromEuler(new Euler(rot[0]! * RAD, rot[1]! * RAD, rot[2]! * RAD, 'XYZ'));
  return new Matrix4().compose(new Vector3(pos[0], pos[1], pos[2]), q, new Vector3(scale[0], scale[1], scale[2])).toArray();
}
