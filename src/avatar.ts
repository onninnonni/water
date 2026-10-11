// Simple animated pirate: boxes for legs, coat, arms, head and a tricorn hat.
import { MeshBuilder, V3 } from './mesh';
import { Mat4, multiply } from './mat4';

export interface AvatarParts { leg: number[]; torso: number[]; arm: number[]; head: number[]; }
export function buildAvatarParts(coat: V3): AvatarParts {
  const SKIN: V3 = [0.72, 0.52, 0.4], BOOT: V3 = [0.07, 0.05, 0.04], PANTS: V3 = [0.32, 0.28, 0.22];
  const leg = new MeshBuilder();
  leg.box(-0.09, 0.09, -0.08, 0.08, -0.62, 0, PANTS);
  leg.box(-0.1, 0.16, -0.09, 0.09, -0.95, -0.62, BOOT);
  const torso = new MeshBuilder();
  torso.box(-0.15, 0.15, -0.25, 0.25, 0.72, 1.56, coat);
  torso.box(-0.16, 0.16, -0.26, 0.26, 0.98, 1.06, [0.08, 0.06, 0.04]);
  torso.box(0.15, 0.16, -0.05, 0.05, 1.0, 1.04, [0.75, 0.62, 0.2]);
  const arm = new MeshBuilder();
  arm.box(-0.07, 0.07, -0.07, 0.07, -0.55, 0.04, coat);
  arm.box(-0.06, 0.06, -0.06, 0.06, -0.68, -0.55, SKIN);
  const head = new MeshBuilder();
  head.box(-0.12, 0.12, -0.11, 0.11, 1.58, 1.84, SKIN);
  head.box(0.12, 0.13, -0.08, -0.02, 1.7, 1.74, [0.05, 0.05, 0.05]);
  head.box(0.12, 0.13, 0.02, 0.08, 1.7, 1.74, [0.05, 0.05, 0.05]);
  head.box(-0.24, 0.2, -0.3, 0.3, 1.84, 1.89, [0.06, 0.05, 0.05]);
  head.box(-0.13, 0.12, -0.14, 0.14, 1.88, 2.02, [0.06, 0.05, 0.05]);
  return { leg: leg.data, torso: torso.data, arm: arm.data, head: head.data };
}

const T = (x: number, y: number, z: number): Mat4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const RY = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
const RX = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
export const RZ = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); };
export const TR = T;

export interface Anim { phase: number; walk: number; wave: number; hold: number; time: number; look: number; }
// Returns [part, matrix] pairs; base = ship model * T(feet) * RZ(yaw).
export function avatarPose(base: Mat4, a: Anim): [keyof AvatarParts, Mat4][] {
  const swing = Math.sin(a.phase) * 0.65 * a.walk;
  const bob = Math.abs(Math.sin(a.phase)) * 0.05 * a.walk;
  const b = multiply(base, T(0, 0, bob));
  const out: [keyof AvatarParts, Mat4][] = [];
  out.push(['leg', multiply(b, multiply(T(0, 0.13, 0.95), RY(-swing)))]);
  out.push(['leg', multiply(b, multiply(T(0, -0.13, 0.95), RY(swing)))]);
  out.push(['torso', b]);
  out.push(['head', multiply(b, multiply(T(0, 0, 1.58), multiply(RY(-a.look * 0.6), T(0, 0, -1.58))))]);
  // Arms: swing while walking, reach forward when holding the wheel or harpoon, wave overhead.
  const hold = a.hold;
  const left = swing * (1 - hold) - hold * 1.25;
  out.push(['arm', multiply(b, multiply(T(0, 0.33, 1.5), RY(left)))]);
  if (a.wave > 0) {
    const w = a.wave;
    out.push(['arm', multiply(b, multiply(T(0, -0.33, 1.5), multiply(RX(w * (2.7 + Math.sin(a.time * 9) * 0.35)), RY(-hold * 1.25 * (1 - w)))))]);
  } else out.push(['arm', multiply(b, multiply(T(0, -0.33, 1.5), RY(-swing * (1 - hold) - hold * 1.25)))]);
  return out;
}
