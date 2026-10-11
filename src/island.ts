// The island: a crescent of cliffs around a sheltered bay, sea stacks across the mouth with an
// arch, a lighthouse on the eastern horn, and channel buoys leading out into the storm.
import { IC, clamp, smoothstep } from './world';
import { MeshBuilder, V3 } from './mesh';

// ---- Noise (deterministic, shared by mesh, collision and baked maps) ----
function hash(ix: number, iy: number) {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
export function vnoise(x: number, y: number) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}
export function fbm(x: number, y: number, oct = 4) {
  let s = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { s += amp * (vnoise(x * f + i * 17.3, y * f - i * 9.1) * 2 - 1); f *= 2.03; amp *= 0.5; }
  return s;
}

const GAP_ANG = -Math.PI / 2, GAP_HALF = 0.62;

// Terrain height (m, sea level 0). Outer side drops in cliffs, inner side slopes to beaches.
export function terrainHeight(x: number, y: number) {
  const dx = x - IC.x, dy = y - IC.y;
  const r = Math.hypot(dx, dy);
  const n1 = fbm(x * 0.005, y * 0.005, 4);
  const d = r + n1 * 48 - IC.R;
  const w = d < 0 ? 175 : 95;
  const t = clamp(1 - Math.abs(d) / w, 0, 1);
  let da = Math.abs(Math.atan2(dy, dx) - GAP_ANG);
  if (da > Math.PI) da = 2 * Math.PI - da;
  const g = smoothstep(GAP_HALF, GAP_HALF + 0.2, da + n1 * 0.06);
  const shape = t * t * (3 - 2 * t) * g;
  const coast = 0.3;
  const depth = 18 + 30 * smoothstep(-60, 60, d);
  if (shape < coast) return -(1 - shape / coast) * depth;
  const u = (shape - coast) / (1 - coast);
  const hills = 30 + 75 * (0.5 + 0.5 * fbm(x * 0.003 + 11, y * 0.003 - 7, 3)) + 10 * fbm(x * 0.02, y * 0.02, 2);
  const prof = Math.pow(u, 1.1) + (Math.pow(u, 0.3) - Math.pow(u, 1.1)) * smoothstep(-20, 20, d);
  return prof * hills;
}

// ---- Sea stacks, arch, lighthouse, buoys ----
export interface Pillar { x: number; y: number; r: number; h: number; }
const PR = 545;
export const PILLARS: Pillar[] = [
  [-113, 40, 13], [-103, 60, 15], [-94, 72, 16], [-86, 68, 15], [-77, 52, 14], [-67, 38, 12],
].map(([a, h, r]) => ({ x: IC.x + PR * Math.cos(a * Math.PI / 180), y: IC.y + PR * Math.sin(a * Math.PI / 180), r, h }));
export const ARCH = { a: 2, b: 3, z0: 50, rise: 10, thick: 6.5 };

const LH_ANG = -42 * Math.PI / 180;
export const LIGHTHOUSE = (() => {
  const x = IC.x + (IC.R + 5) * Math.cos(LH_ANG), y = IC.y + (IC.R + 5) * Math.sin(LH_ANG);
  const base = Math.max(terrainHeight(x, y), 2);
  return { x, y, base, lamp: base + 28 };
})();

export type BuoyKind = 0 | 1 | 2; // red, green, yellow
export interface Buoy { x: number; y: number; kind: BuoyKind; phase: number; }
export const BUOYS: Buoy[] = [
  ...[-330, -440, -650, -770, -900].flatMap((y, i) => [
    { x: IC.x - 18, y: IC.y + y, kind: 0 as BuoyKind, phase: i * 0.7 },
    { x: IC.x + 18, y: IC.y + y, kind: 1 as BuoyKind, phase: i * 0.7 + 0.35 },
  ]),
  ...Array.from({ length: 8 }, (_, i) => {
    const a = (i * 45 + 12) * Math.PI / 180;
    return { x: IC.x + 1500 * Math.cos(a), y: IC.y + 1500 * Math.sin(a), kind: 2 as BuoyKind, phase: i * 1.3 };
  }),
];

// Height of anything solid: terrain plus sea stacks (used for collision and the shore map).
export function obstacleHeight(x: number, y: number) {
  let h = terrainHeight(x, y);
  for (const p of PILLARS) if (Math.hypot(x - p.x, y - p.y) < p.r) h = Math.max(h, p.h);
  return h;
}

// Baked shore map: (h + 20) / 40 in an R8 texture, used by the water shader for shallows and surf.
export const SHORE = { min: -800, size: 1600, res: 512 };
export function buildShoreMap(): Uint8Array {
  const { min, size, res } = SHORE;
  const out = new Uint8Array(res * res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const x = IC.x + min + (i + 0.5) / res * size, y = IC.y + min + (j + 0.5) / res * size;
    out[j * res + i] = Math.round(clamp((obstacleHeight(x, y) + 20) / 40, 0, 1) * 255);
  }
  return out;
}

// ---- Meshes ----
const lerp3 = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

function terrainColour(a: V3, b: V3, c: V3, inner: boolean): V3 {
  const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
  const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
  const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
  const slope = Math.abs(nz) / (Math.hypot(nx, ny, nz) || 1);
  const h = (a[2] + b[2] + c[2]) / 3;
  const cx = (a[0] + b[0] + c[0]) / 3, cy = (a[1] + b[1] + c[1]) / 3;
  const n = vnoise(cx * 0.05, cy * 0.05);
  if (h < 0) return [0.1, 0.11, 0.1];
  if (slope < 0.72) {
    const rock = lerp3([0.17, 0.16, 0.15], [0.27, 0.25, 0.22], n);
    return h < 4 ? lerp3(rock, [0.06, 0.07, 0.07], 0.6) : rock;
  }
  if (h < 3.5 && inner) return lerp3([0.42, 0.38, 0.29], [0.5, 0.45, 0.35], n);
  if (h < 2.5) return [0.12, 0.12, 0.11];
  const grass = lerp3([0.11, 0.15, 0.08], [0.18, 0.21, 0.11], n);
  return h > 85 ? lerp3(grass, [0.25, 0.24, 0.21], smoothstep(85, 110, h) * 0.7) : grass;
}

function addTerrain(m: MeshBuilder) {
  const N = 200, span = 1560, x0 = IC.x - span / 2, y0 = IC.y - span / 2, step = span / N;
  const H = new Float32Array((N + 1) * (N + 1));
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) H[j * (N + 1) + i] = terrainHeight(x0 + i * step, y0 + j * step);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const h00 = H[j * (N + 1) + i], h10 = H[j * (N + 1) + i + 1], h01 = H[(j + 1) * (N + 1) + i], h11 = H[(j + 1) * (N + 1) + i + 1];
    if (Math.max(h00, h10, h01, h11) < -6) continue;
    const xa = x0 + i * step, ya = y0 + j * step;
    const p00: V3 = [xa, ya, h00], p10: V3 = [xa + step, ya, h10], p01: V3 = [xa, ya + step, h01], p11: V3 = [xa + step, ya + step, h11];
    const inner = Math.hypot(xa - IC.x, ya - IC.y) < IC.R;
    m.tri(p00, p10, p11, terrainColour(p00, p10, p11, inner));
    m.tri(p00, p11, p01, terrainColour(p00, p11, p01, inner));
  }
}

function rockCol(z: number, top: number, seed: number): V3 {
  if (z < 3) return [0.05, 0.065, 0.06];                       // wet, weedy base
  if (z > top - 4) return [0.16, 0.2, 0.11];                    // grassy crown
  const band = vnoise(z * 0.18 + seed, seed * 2.3);
  const c = lerp3([0.14, 0.135, 0.13], [0.25, 0.23, 0.2], band);
  return z > top * 0.8 ? lerp3(c, [0.42, 0.41, 0.38], 0.35) : c; // bird-streaked upper rock
}

function addPillar(m: MeshBuilder, p: Pillar, seed: number) {
  const seg = 18, rings = 18, zb = -18;
  const pos = (i: number, j: number): V3 => {
    const a = (i / seg) * Math.PI * 2, zt = j / rings, z = zb + (p.h - zb) * zt;
    const ca = Math.cos(a), sa = Math.sin(a);
    // Irregular, fractured column: big lumps, slanted ledges, a lean, and a narrower crown.
    const lump = fbm(ca * 1.4 + seed * 3.1 + z * 0.035, sa * 1.4 - seed + z * 0.05, 3);
    const ledge = Math.max(0, Math.sin(z * 0.22 + ca * 1.7 + seed * 2)) * 0.12;
    const rr = p.r * (1.05 - 0.38 * Math.pow(Math.max(zt, 0), 1.4)) * (1 + 0.42 * lump + ledge);
    const lean = (zt * zt) * p.r * 0.35;
    return [p.x + ca * rr + Math.cos(seed * 5) * lean, p.y + sa * rr + Math.sin(seed * 5) * lean, z];
  };
  for (let j = 0; j < rings; j++) for (let i = 0; i < seg; i++) {
    const a = pos(i, j), b = pos(i + 1, j), c = pos(i + 1, j + 1), d = pos(i, j + 1);
    m.quad(a, b, c, d, rockCol((a[2] + c[2]) / 2, p.h, seed));
  }
  let tx = 0, ty = 0;
  for (let i = 0; i < seg; i++) { const q = pos(i, rings); tx += q[0] / seg; ty += q[1] / seg; }
  const top: V3 = [tx, ty, p.h + 2];
  for (let i = 0; i < seg; i++) m.tri(pos(i, rings), pos(i + 1, rings), top, [0.15, 0.19, 0.1]);
}

function addArch(m: MeshBuilder) {
  const A = PILLARS[ARCH.a], B = PILLARS[ARCH.b];
  const curve = (t: number): V3 => {
    const z = ARCH.z0 + 4 * ARCH.rise * t * (1 - t);
    return [A.x + (B.x - A.x) * t, A.y + (B.y - A.y) * t, z];
  };
  const L = 20, S = 10;
  const ax = B.x - A.x, ay = B.y - A.y, al = Math.hypot(ax, ay);
  const sx = -ay / al, sy = ax / al;
  const ring = (i: number, k: number): V3 => {
    const t = i / L, c = curve(t);
    const a = (k / S) * Math.PI * 2;
    const r = ARCH.thick * (0.85 + 0.3 * vnoise(i * 0.7, k * 0.9)) * (1 + 0.35 * Math.abs(t - 0.5) * 2);
    return [c[0] + sx * Math.cos(a) * r * 1.3, c[1] + sy * Math.cos(a) * r * 1.3, c[2] + Math.sin(a) * r];
  };
  for (let i = 0; i < L; i++) for (let k = 0; k < S; k++) {
    const a = ring(i, k), b = ring(i + 1, k), c = ring(i + 1, k + 1), d = ring(i, k + 1);
    const up = Math.sin(((k + 0.5) / S) * Math.PI * 2) > 0.6;
    m.quad(a, b, c, d, up ? [0.15, 0.19, 0.1] : rockCol((a[2] + c[2]) / 2, 80, 3));
  }
}

function addLighthouse(m: MeshBuilder) {
  const { x, y, base } = LIGHTHOUSE;
  m.cyl(x, y, base - 6, base + 2.5, 4.6, 4.3, 14, [0.32, 0.31, 0.29]);
  m.cyl(x, y, base + 2.5, base + 26, 3.2, 2.4, 14, (z) => (Math.floor((z - base) / 4.7) % 2 === 0 ? [0.82, 0.81, 0.78] : [0.62, 0.1, 0.08]));
  m.cyl(x, y, base + 26, base + 26.6, 3.5, 3.5, 14, [0.1, 0.1, 0.1]);
  m.cyl(x, y, base + 26.6, base + 29.2, 2.0, 2.0, 12, [0.14, 0.17, 0.18], false);
  m.cyl(x, y, base + 29.2, base + 31.2, 2.5, 0.2, 12, [0.55, 0.08, 0.06]);
  // Keeper's cottage beside the tower.
  const hx = x - 14, hy = y + 6, hz = Math.max(terrainHeight(hx, hy), base - 3);
  m.box(hx - 5, hx + 5, hy - 3.5, hy + 3.5, hz - 3, hz + 3.2, [0.7, 0.68, 0.62]);
  m.quad([hx - 5.4, hy - 3.9, hz + 3.2], [hx + 5.4, hy - 3.9, hz + 3.2], [hx + 5.4, hy, hz + 6], [hx - 5.4, hy, hz + 6], [0.25, 0.12, 0.1]);
  m.quad([hx - 5.4, hy + 3.9, hz + 3.2], [hx + 5.4, hy + 3.9, hz + 3.2], [hx + 5.4, hy, hz + 6], [hx - 5.4, hy, hz + 6], [0.25, 0.12, 0.1]);
  m.box(hx + 1.5, hx + 3.0, hy - 3.6, hy - 3.45, hz + 0.2, hz + 1.8, [2.9, 2.6, 1.9]); // lit window
}

export function buildLand(): number[] {
  const m = new MeshBuilder();
  addTerrain(m);
  PILLARS.forEach((p, i) => addPillar(m, p, i * 1.7));
  addArch(m);
  addLighthouse(m);
  return m.data;
}

export const BUOY_COLS: V3[] = [[0.65, 0.08, 0.06], [0.06, 0.45, 0.15], [0.8, 0.62, 0.05]];
export function buildBuoy(kind: BuoyKind): number[] {
  const m = new MeshBuilder();
  const col = BUOY_COLS[kind];
  m.cyl(0, 0, -1.2, 0.6, 1.1, 1.1, 12, col);
  m.cyl(0, 0, 0.6, 0.8, 1.1, 0.5, 12, [0.12, 0.12, 0.12]);
  for (let i = 0; i < 3; i++) {
    const a = i * Math.PI * 2 / 3;
    m.box(Math.cos(a) * 0.45 - 0.06, Math.cos(a) * 0.45 + 0.06, Math.sin(a) * 0.45 - 0.06, Math.sin(a) * 0.45 + 0.06, 0.8, 3.0, col);
  }
  if (kind === 0) m.cyl(0, 0, 2.6, 3.4, 0.55, 0.55, 10, col);
  else if (kind === 1) m.cyl(0, 0, 2.6, 3.5, 0.6, 0.05, 10, col);
  else { m.cyl(0, 0, 2.6, 3.1, 0.05, 0.55, 10, [0.1, 0.1, 0.1]); m.cyl(0, 0, 3.1, 3.6, 0.55, 0.05, 10, [0.1, 0.1, 0.1]); }
  return m.data;
}
