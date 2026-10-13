// The Great Octopus that seizes ships at the edge of the Eye: meshes, and the shape of its
// tentacles over a ship's side, worked out in that ship's local coordinates (x forward, y port,
// z up) so every client draws the same thing from the same few numbers.
import { MeshBuilder, V3 } from './mesh';
import { halfBeam, gunZ, deckZ } from './ship';

export const TENTACLES = [-13, -8.5, -4, 0.5, 5, 9.5]; // where each one comes over the rail (x)
export const OCTO_HP = 5;                                // flasks of poison to make it let go
export const SLAM_TIME = 1.2;                            // rise, then strike

export function buildMantleMesh(): number[] {
  const m = new MeshBuilder();
  const SKIN = (z: number): V3 => (z > 0.4 ? [0.42, 0.12, 0.16] : z > -0.2 ? [0.5, 0.17, 0.18] : [0.62, 0.36, 0.34]);
  m.sphere([0, 0, 0], 1, 0.85, 1.15, 16, SKIN);
  for (const s of [1, -1]) {
    m.sphere([0.62, s * 0.55, -0.35], 0.24, 0.24, 0.24, 10, [2.85, 2.75, 2.2]);
    m.box(0.83, 0.88, s * 0.5 - 0.02, s * 0.5 + 0.02, -0.5, -0.2, [0.02, 0.02, 0.02]);
  }
  // Mottled spots.
  for (let k = 0; k < 14; k++) {
    const a = k * 2.4, b = 0.5 + (k % 5) * 0.12;
    m.sphere([Math.cos(a) * 0.8 * Math.sin(b * 2), Math.sin(a) * 0.68 * Math.sin(b * 2), 0.5 + Math.cos(b * 2) * 0.4], 0.12, 0.12, 0.05, 6, [0.28, 0.07, 0.1]);
  }
  return m.data;
}
// One tentacle segment: a unit ellipsoid along x, dark on top with pale suckers beneath.
export function buildTentacleMesh(): number[] {
  const m = new MeshBuilder();
  m.sphere([0, 0, 0], 1, 1, 1, 10, (z) => (z > 0.1 ? [0.44, 0.13, 0.17] : [0.75, 0.5, 0.45]));
  return m.data;
}

const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function catmull(p0: V3, p1: V3, p2: V3, p3: V3, t: number): V3 {
  const t2 = t * t, t3 = t2 * t;
  const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
  return [f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1]), f(p0[2], p1[2], p2[2], p3[2])];
}

export interface OctoView { side: number; t: number; recoil: number[]; release: number; slam: { i: number; x: number; y: number; p: number } | null; }

export function mantleLocal(o: OctoView): V3 {
  const s = o.side, x = -2;
  return [x, s * (halfBeam(x) + 8.5), 0.6 + 0.6 * Math.sin(o.t * 0.7) - o.release * 14];
}

// Sample points (local) along tentacle i, from the mantle to its tip on the deck.
export function tentaclePath(o: OctoView, i: number, n = 16): { p: V3; r: number }[] {
  const s = o.side, x = TENTACLES[i], hb = halfBeam(x), t = o.t, w = Math.sin(t * 1.6 + i * 1.3);
  const M = mantleLocal(o);
  const base: V3 = [M[0] + (x - M[0]) * 0.35, M[1] - s * 2, M[2] - 2];
  const water: V3 = [x, s * (hb + 1.8), -0.8];
  const rail: V3 = [x + 0.2 * w, s * (hb + 0.15), gunZ(x) + 0.75];
  let tip: V3 = [x + 1.2 * w, s * (hb - 2.6 - 0.5 * Math.sin(t * 1.1 + i)), deckZ(x, 0) + 0.35];
  let curl: V3 = [tip[0] + 1.5 * Math.cos(t + i), tip[1] - s * 1.2, tip[2] + 0.3];
  const sl = o.slam;
  if (sl && sl.i === i) {
    // Rearing up over the deck, then crashing down where someone stands.
    const hit: V3 = [sl.x, sl.y, deckZ(sl.x, sl.y) + 0.3];
    const up: V3 = [(x + sl.x) / 2, (rail[1] + sl.y) / 2, deckZ(x, 0) + 8];
    const rest = tip;
    tip = sl.p < 0.75 ? lerp3(tip, up, Math.min(1, sl.p / 0.5)) : sl.p <= 1 ? lerp3(up, hit, (sl.p - 0.75) / 0.25) : lerp3(hit, rest, Math.min(1, (sl.p - 1) / 0.6));
    curl = [tip[0] + (tip[0] - rail[0]) * 0.2, tip[1] + (tip[1] - rail[1]) * 0.2, tip[2] - 0.5];
  }
  // Poisoned: it recoils off the deck back over the side for a while.
  const rc = Math.max(o.recoil[i] ?? 0, o.release);
  if (rc > 0) {
    const k = Math.min(1, rc);
    const off: V3 = [x, s * (hb + 3), -3 - 10 * o.release];
    tip = lerp3(tip, off, k); curl = lerp3(curl, [off[0], off[1] + s * 2, off[2] - 2], k);
  }
  const pts = [base, water, rail, tip, curl];
  const out: { p: V3; r: number }[] = [];
  for (let k = 0; k < n; k++) {
    const u = k / (n - 1) * (pts.length - 1), j = Math.min(pts.length - 2, Math.floor(u)), f = u - j;
    const p = catmull(pts[Math.max(0, j - 1)], pts[j], pts[j + 1], pts[Math.min(pts.length - 1, j + 2)], f);
    out.push({ p, r: 1.25 * (1 - k / n) + 0.18 });
  }
  return out;
}
