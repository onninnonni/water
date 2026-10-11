// Simple animated pirate: boxes for legs, coat, arms, head and a tricorn hat, plus a cutlass.
import { MeshBuilder, V3 } from './mesh';
import { Mat4, multiply } from './mat4';

export interface AvatarParts { leg: number[]; torso: number[]; arm: number[]; head: number[]; sword: number[]; }
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
  // Cutlass held in the hand, blade pointing forward along +x of the hand frame.
  const sword = new MeshBuilder();
  sword.box(-0.03, 0.03, -0.03, 0.03, -0.06, 0.06, [0.2, 0.12, 0.06]);
  sword.box(0.02, 0.05, -0.09, 0.09, -0.02, 0.02, [0.6, 0.5, 0.2]);
  sword.quad([0.05, 0, -0.025], [0.85, 0, -0.03], [0.9, 0, 0.02], [0.05, 0, 0.035], [0.78, 0.8, 0.84]);
  return { leg: leg.data, torso: torso.data, arm: arm.data, head: head.data, sword: sword.data };
}

const T = (x: number, y: number, z: number): Mat4 => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const RY = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
const RX = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
export const RZ = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]); };
export const TR = T;
const M = (...ms: Mat4[]) => ms.reduce((a, b) => multiply(a, b));

export interface Anim {
  phase: number; walk: number; wave: number; hold: number; time: number; look: number;
  swing: number;   // 0..1 progress of a cutlass swing (0 = idle)
  climb: number;   // 0 or 1
  down: number;    // 0..1 knocked down
  sword: boolean;
}
// Returns [part, matrix] pairs; base = ship model * T(feet) * RZ(yaw).
export function avatarPose(base: Mat4, a: Anim): [keyof AvatarParts, Mat4][] {
  const out: [keyof AvatarParts, Mat4][] = [];
  // Knocked down: lying on the back.
  const b0 = a.down > 0 ? M(base, T(0, 0, 0.18 * a.down), RY(-a.down * Math.PI / 2), T(-0.9 * a.down, 0, 0)) : base;
  const climbSwing = a.climb ? Math.sin(a.phase) : 0;
  const swing = a.climb ? 0 : Math.sin(a.phase) * 0.65 * a.walk;
  const bob = Math.abs(Math.sin(a.phase)) * 0.05 * a.walk;
  const b = M(b0, T(0, 0, bob));
  out.push(['leg', M(b, T(0, 0.13, 0.95), RY(-swing - (a.climb ? 0.6 + climbSwing * 0.5 : 0)))]);
  out.push(['leg', M(b, T(0, -0.13, 0.95), RY(swing - (a.climb ? 0.6 - climbSwing * 0.5 : 0)))]);
  out.push(['torso', b]);
  out.push(['head', M(b, T(0, 0, 1.58), RY(-a.look * 0.6), T(0, 0, -1.58))]);
  const hold = a.hold;
  // Left arm.
  const leftA = a.climb ? -2.6 + climbSwing * 0.5 : swing * (1 - hold) - hold * 1.25;
  out.push(['arm', M(b, T(0, 0.33, 1.5), RY(leftA))]);
  // Right arm: wave, climb, a cutlass swing, or swinging at the side.
  let right: Mat4;
  if (a.wave > 0) right = M(b, T(0, -0.33, 1.5), RX(a.wave * (2.7 + Math.sin(a.time * 9) * 0.35)));
  else if (a.climb) right = M(b, T(0, -0.33, 1.5), RY(-2.6 - climbSwing * 0.5));
  else if (a.swing > 0) {
    const s = a.swing, raise = s < 0.35 ? s / 0.35 : 1 - (s - 0.35) / 0.65;
    right = M(b, T(0, -0.33, 1.5), RZ(0.6 - s * 1.4), RY(-1.2 - raise * 1.4 + (1 - raise) * 0.4));
  } else right = M(b, T(0, -0.33, 1.5), RY(-swing * (1 - hold) - hold * 1.25 - (a.sword ? 0.35 : 0)));
  out.push(['arm', right]);
  if (a.sword && !a.climb && a.wave <= 0 && hold === 0) out.push(['sword', M(right, T(0, 0, -0.62), RY(Math.PI / 2 - 0.2))]);
  return out;
}
