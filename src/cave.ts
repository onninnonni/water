// The Leviathan's lair: an air pocket under a rock dome, deep below the sea. Built around its own
// origin (0, 0 at the centre of the still water, z = 0 the waterline); the exit tunnel leaves
// along +x, sloping down into the flooded dark.
import { MeshBuilder, V3 } from './mesh';

// Far from the sea anyone sails, so nothing else ever shares these coordinates.
export const CAVE = { x: -30000, y: 30000, R: 125, H: 62 };
export const CAVE_SHIP = { x: -30, y: 0, psi: 0 }; // where the ghost ship floats, bow to the tunnel

const hash = (x: number, y: number) => { const h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return h - Math.floor(h); };
function noise(x: number, y: number) {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
const mixc = (a: V3, b: V3, t: number): V3 => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

export interface CaveMesh { data: number[]; crystals: { p: V3; col: V3 }[]; }

export function buildCave(): CaveMesh {
  const m = new MeshBuilder();
  const { R, H } = CAVE;
  const ROCK: V3 = [0.16, 0.15, 0.14], WET: V3 = [0.06, 0.07, 0.08], MOSS: V3 = [0.07, 0.13, 0.11];
  // Dome: a lumpy, flattened hemisphere that dips below the waterline; the tunnel mouth (+x) is
  // pushed out into a dark arch.
  const NT = 22, NP = 64;
  const pt = (i: number, j: number): V3 => {
    const th = (i / NT) * Math.PI * 0.6, ph = (j / NP) * Math.PI * 2;
    const n = noise(Math.cos(ph) * 3 + 10, Math.sin(ph) * 3 + i * 0.45) * 0.28 + noise(j * 0.9, i * 0.9) * 0.08;
    let r = R * (0.86 + n), z = H * Math.cos(th) * (0.85 + n * 0.6) - 10 * Math.max(0, i - NT * 0.82);
    const mouth = Math.max(0, Math.cos(ph) - 0.93) / 0.07; // 0..1 near +x
    if (i > NT * 0.62) { r += mouth * R * 0.9; z -= mouth * 6; }
    return [Math.cos(ph) * Math.sin(Math.max(th, 0.02)) * r * 1.0 / Math.max(Math.sin(Math.PI * 0.6), 0.5), Math.sin(ph) * Math.sin(Math.max(th, 0.02)) * r / Math.max(Math.sin(Math.PI * 0.6), 0.5), z];
  };
  for (let i = 0; i < NT; i++) for (let j = 0; j < NP; j++) {
    const a = pt(i, j), b = pt(i, j + 1), c = pt(i + 1, j + 1), d = pt(i + 1, j);
    const zc = (a[2] + c[2]) / 2, n = noise(j * 0.7, i * 0.7);
    let col = mixc(ROCK, MOSS, n * 0.6);
    if (zc < 6) col = mixc(WET, col, Math.max(0, zc) / 6);
    col = mixc(col, [0.1, 0.1, 0.11], 0.3 * hash(i, j));
    m.quad(a, b, c, d, col);
  }
  // The flooded tunnel: a dark bore sloping down and away beyond the mouth.
  const T0: V3 = [R * 1.45, 0, 0], T1: V3 = [R * 2.6, 0, -40];
  m.tube(T0, T1, 22, 16, 14, [0.04, 0.045, 0.05], false);
  // Stalactites hanging from the vault.
  for (let k = 0; k < 46; k++) {
    const a = hash(k, 3) * Math.PI * 2, rr = Math.sqrt(hash(k, 7)) * R * 0.78;
    const x = Math.cos(a) * rr, y = Math.sin(a) * rr;
    const top = H * Math.sqrt(Math.max(0, 1 - (rr / (R * 0.95)) ** 2)) * 0.95 + 4;
    const len = 5 + hash(k, 11) * 16;
    m.tube([x, y, top], [x + (hash(k, 13) - 0.5) * 2, y, top - len], 1.2 + hash(k, 5) * 2.2, 0.05, 6, mixc(ROCK, [0.22, 0.2, 0.18], hash(k, 17)));
  }
  // Boulders breaking the still water near the walls.
  for (let k = 0; k < 14; k++) {
    const a = hash(k, 23) * Math.PI * 2, rr = R * (0.72 + hash(k, 29) * 0.2);
    if (Math.cos(a) > 0.85) continue; // keep the way to the tunnel clear
    m.sphere([Math.cos(a) * rr, Math.sin(a) * rr, -1], 5 + hash(k, 31) * 9, 5 + hash(k, 37) * 8, 3 + hash(k, 41) * 5, 8, (z) => (z > 0.5 ? ROCK : WET));
  }
  // Still black water, faintly lit from within.
  const WATER: V3 = [2.012, 2.055, 2.07];
  const NW = 48;
  for (let j = 0; j < NW; j++) {
    const a0 = (j / NW) * Math.PI * 2, a1 = ((j + 1) / NW) * Math.PI * 2, rr = R * 1.5;
    m.tri([0, 0, 0], [Math.cos(a0) * rr, Math.sin(a0) * rr, 0], [Math.cos(a1) * rr, Math.sin(a1) * rr, 0], WATER);
  }
  // Glowing crystal clusters on the walls and the vault.
  const crystals: { p: V3; col: V3 }[] = [];
  const CRY: V3[] = [[0.25, 0.95, 0.85], [0.55, 0.45, 1.0], [0.3, 0.75, 1.0]];
  for (let k = 0; k < 34; k++) {
    const a = hash(k, 43) * Math.PI * 2, up = hash(k, 47);
    const rr = R * (0.8 - up * 0.45), z = 2 + up * H * 0.8;
    const base: V3 = [Math.cos(a) * rr, Math.sin(a) * rr, z];
    const col = CRY[k % 3];
    for (let q = 0; q < 4; q++) {
      const dx = -Math.cos(a) * (0.4 + hash(k, q) * 0.6), dy = -Math.sin(a) * (0.4 + hash(q, k) * 0.6), dz = (hash(k + q, 5) - 0.3) * 0.9;
      const len = 1.5 + hash(k, q + 9) * 3.5;
      m.tube(base, [base[0] + dx * len, base[1] + dy * len, base[2] + dz * len], 0.35, 0.02, 5, [2 + col[0] * 0.8, 2 + col[1] * 0.8, 2 + col[2] * 0.8]);
    }
    crystals.push({ p: base, col });
  }
  return { data: m.data, crystals };
}

// Recolour a ship mesh as the drowned ghost ship: bleached, sea-green timber; emissive parts
// (lanterns) burn green.
export function ghostify(data: number[], glow = false): number[] {
  const out = data.slice();
  for (let i = 3; i < out.length; i += 6) {
    const r = out[i], g = out[i + 1], b = out[i + 2];
    if (r > 1.5) { out[i] = 2.25; out[i + 1] = 2.95; out[i + 2] = 2.6; continue; }
    const l = 0.3 * r + 0.59 * g + 0.11 * b;
    if (glow) { out[i] = 2 + 0.06 + l * 0.12; out[i + 1] = 2 + 0.2 + l * 0.32; out[i + 2] = 2 + 0.16 + l * 0.22; continue; }
    out[i] = 0.1 + l * 0.55; out[i + 1] = 0.15 + l * 0.68; out[i + 2] = 0.15 + l * 0.66;
  }
  return out;
}
