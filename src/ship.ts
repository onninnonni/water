// A 40 m pirate ship. Local frame: x forward, y to port (left), z up, z = 0 at the waterline.
import { MeshBuilder, V3 } from './mesh';
import { sampleWater, newSample } from './waves';
import { terrainHeight, PILLARS } from './island';
import { clamp, smoothstep, WIND_DIR } from './world';

export const STERN = -18, BOW = 18;
export const MAIN_DECK = 4.0, QUARTER_DECK = 6.5, FORE_DECK = 5.5;

export function halfBeam(x: number) {
  const t = clamp((x - STERN) / (BOW - STERN), 0, 1);
  return t < 0.45 ? 5 * (0.82 + 0.18 * Math.sin(Math.PI / 2 * t / 0.45)) : 5 * Math.pow(Math.max(Math.cos((t - 0.45) / 0.55 * Math.PI / 2), 0), 0.7);
}
const keelZ = (x: number) => {
  const t = clamp((x - STERN) / (BOW - STERN), 0, 1);
  return -4.2 * (t > 0.78 ? 1 - Math.pow((t - 0.78) / 0.22, 1.6) * 0.75 : 1) * (t < 0.05 ? 0.75 + 5 * t : 1);
};
export const gunZ = (x: number) => 5.2 + 2.5 * smoothstep(-10.2, -11.4, x) + 1.5 * smoothstep(13, 14.2, x) + 0.8 * smoothstep(15, 18, x);
const sideDeck = (x: number) => (x < -11 ? QUARTER_DECK : x > 13.5 ? FORE_DECK : MAIN_DECK);

// ---- Deck layout (walkable) ----
export function deckZ(x: number, y: number) {
  if (x < -11) return QUARTER_DECK;
  if (x < -9 && Math.abs(y) < 1.6) return MAIN_DECK + 2.5 * (-9 - x) / 2;
  if (x > 13.5) return FORE_DECK;
  if (x > 12 && Math.abs(y) < 1.4) return MAIN_DECK + 1.5 * (x - 12) / 1.5;
  return MAIN_DECK;
}
export const MASTS = [
  { x: 9.5, base: MAIN_DECK, top: 37, r: 0.45, yards: [{ z: 18.5, w: 15, h: 12 }, { z: 28.5, w: 12, h: 9.5 }] },
  { x: 0, base: MAIN_DECK, top: 42, r: 0.5, yards: [{ z: 20, w: 17, h: 13 }, { z: 31, w: 13.5, h: 10.5 }] },
  { x: -15.5, base: QUARTER_DECK, top: 32, r: 0.38, yards: [{ z: 19, w: 11, h: 9 }, { z: 27, w: 9, h: 7 }] },
];
export const CANNONS = [-5, 1, 7].flatMap((x) => [1, -1].map((side) => ({ x, side, y: side * (halfBeam(x) - 1.2), z: MAIN_DECK })));
export const HELM = { x: -12.4, z: QUARTER_DECK + 0.8, stand: [-14.4, 0] as [number, number] };
export const HARPOON = { x: 15.4, y: 0, z: FORE_DECK + 1.25 };
const OBST = [
  ...MASTS.map((m) => ({ x: m.x, y: 0, r: m.r + 0.35 })),
  { x: HELM.x, y: 0, r: 0.55 }, { x: -7, y: 0, r: 0.75 }, { x: HARPOON.x, y: 0, r: 0.6 },
];

// Move a walker on the deck: stay inside the bulwarks, can step down but not climb walls, slide off obstacles.
export function moveOnDeck(x: number, y: number, dx: number, dy: number): [number, number] {
  const z0 = deckZ(x, y);
  const tryMove = (nx: number, ny: number) => {
    nx = clamp(nx, STERN + 0.8, 16.6);
    const lim = Math.max(halfBeam(nx) - 0.75, 0.25);
    ny = clamp(ny, -lim, lim);
    if (deckZ(nx, ny) - z0 > 0.55) return null;
    for (const o of OBST) {
      const ox = nx - o.x, oy = ny - o.y, d = Math.hypot(ox, oy);
      if (d < o.r) { if (d < 1e-4) return null; nx = o.x + ox / d * o.r; ny = o.y + oy / d * o.r; }
    }
    return [nx, ny] as [number, number];
  };
  return tryMove(x + dx, y + dy) ?? tryMove(x + dx, y) ?? tryMove(x, y + dy) ?? [x, y];
}

// Climbing: a walker goes up the aft face of a mast from the deck to near its top.
export const climbX = (i: number) => MASTS[i].x - MASTS[i].r - 0.4;
export const climbTop = (i: number) => MASTS[i].top - 1.6;
// Rail point on the main deck for a gangplank, just inside the bulwark.
export const railPoint = (x: number, side: number): [number, number] => [clamp(x, -8.5, 10.5), side * (halfBeam(clamp(x, -8.5, 10.5)) - 0.75)];
export function hullContains(l: V3, pad = 0): boolean {
  if (l[0] < STERN - pad || l[0] > BOW + pad) return false;
  return Math.abs(l[1]) < halfBeam(l[0]) + pad && l[2] > -4.5 - pad && l[2] < gunZ(l[0]) + 1.2 + pad;
}
export const HULL_CIRCLES = [-14, -7, 0, 7, 14].map((x) => ({ x, r: halfBeam(x) + 0.4 }));

// ---- Meshes ----
const WOOD_DARK: V3 = [0.22, 0.14, 0.08], OCHRE: V3 = [0.48, 0.34, 0.13], WALE: V3 = [0.06, 0.05, 0.04];
const HULL: V3 = [0.27, 0.17, 0.1], TAR: V3 = [0.1, 0.08, 0.06], CAP: V3 = [0.34, 0.23, 0.13];
const INNER: V3 = [0.38, 0.28, 0.18], PLANK: V3 = [0.42, 0.33, 0.22], ROPE: V3 = [0.12, 0.1, 0.08], MAST: V3 = [0.3, 0.2, 0.12];

export function buildShipMesh(): number[] {
  const m = new MeshBuilder();
  const xs: number[] = [];
  for (let x = STERN; x <= BOW + 1e-6; x += 1) xs.push(x);
  const side = (x: number, s: number) => {
    const w = Math.max(halfBeam(x), 0.03), g = gunZ(x), k = keelZ(x);
    return {
      gun: [x, s * w * 0.95, g] as V3, b1: [x, s * w * 0.98, 4.8] as V3, b2: [x, s * w, 3.6] as V3, b3: [x, s * w, 3.2] as V3,
      wl: [x, s * w * 0.97, 0.6] as V3, bilge: [x, s * w * 0.6, k * 0.8] as V3, keel: [x, 0, k] as V3,
      capIn: [x, s * Math.max(w * 0.95 - 0.3, 0), g] as V3, inBot: [x, s * Math.max(w * 0.95 - 0.3, 0), sideDeck(x)] as V3,
    };
  };
  for (let i = 0; i < xs.length - 1; i++) for (const s of [1, -1]) {
    const a = side(xs[i], s), b = side(xs[i + 1], s);
    m.quad(a.gun, b.gun, b.b1, a.b1, WOOD_DARK);
    m.quad(a.b1, b.b1, b.b2, a.b2, OCHRE);
    m.quad(a.b2, b.b2, b.b3, a.b3, WALE);
    m.quad(a.b3, b.b3, b.wl, a.wl, HULL);
    m.quad(a.wl, b.wl, b.bilge, a.bilge, TAR);
    m.quad(a.bilge, b.bilge, b.keel, a.keel, TAR);
    m.quad(a.gun, b.gun, b.capIn, a.capIn, CAP);
    m.quad(a.capIn, b.capIn, b.inBot, a.inBot, INNER);
  }
  // Decks.
  const platform = (x0: number, x1: number, z: number) => {
    for (let x = x0; x < x1 - 1e-6; x += 0.5) {
      const xa = x, xb = Math.min(x + 0.5, x1);
      const wa = Math.max(halfBeam(xa) * 0.95 - 0.3, 0), wb = Math.max(halfBeam(xb) * 0.95 - 0.3, 0);
      m.quad([xa, -wa, z], [xb, -wb, z], [xb, wb, z], [xa, wa, z], PLANK);
    }
  };
  platform(-11, 13.5, MAIN_DECK); platform(STERN, -11, QUARTER_DECK); platform(13.5, BOW, FORE_DECK);
  for (let x = -10.5; x < 13; x += 1.6) m.box(x, x + 0.02, -halfBeam(x) * 0.9, halfBeam(x) * 0.9, MAIN_DECK, MAIN_DECK + 0.005, [0.3, 0.23, 0.15]);
  // Bulkheads and stairs up to the quarterdeck and forecastle.
  const qw = halfBeam(-11) * 0.95 - 0.3, fw = halfBeam(13.5) * 0.95 - 0.3;
  m.quad([-11, 1.6, MAIN_DECK], [-11, qw, MAIN_DECK], [-11, qw, QUARTER_DECK], [-11, 1.6, QUARTER_DECK], INNER);
  m.quad([-11, -1.6, MAIN_DECK], [-11, -qw, MAIN_DECK], [-11, -qw, QUARTER_DECK], [-11, -1.6, QUARTER_DECK], INNER);
  m.quad([13.5, 1.4, MAIN_DECK], [13.5, fw, MAIN_DECK], [13.5, fw, FORE_DECK], [13.5, 1.4, FORE_DECK], INNER);
  m.quad([13.5, -1.4, MAIN_DECK], [13.5, -fw, MAIN_DECK], [13.5, -fw, FORE_DECK], [13.5, -1.4, FORE_DECK], INNER);
  for (let i = 0; i < 6; i++) m.box(-9 - (i + 1) * 2 / 6, -9 - i * 2 / 6, -1.5, 1.5, MAIN_DECK, MAIN_DECK + (i + 1) * 2.5 / 6, PLANK, [0.46, 0.36, 0.24]);
  for (let i = 0; i < 4; i++) m.box(12 + i * 1.5 / 4, 12 + (i + 1) * 1.5 / 4, -1.3, 1.3, MAIN_DECK, MAIN_DECK + (i + 1) * 1.5 / 4, PLANK, [0.46, 0.36, 0.24]);
  for (const s of [1, -1]) {
    m.tube([-11, s * qw, QUARTER_DECK + 1], [-11, s * 1.6, QUARTER_DECK + 1], 0.06, 0.06, 5, CAP);
    for (let y = 1.6; y < qw; y += 0.9) m.tube([-11, s * y, QUARTER_DECK], [-11, s * y, QUARTER_DECK + 1], 0.05, 0.05, 4, CAP);
  }
  // Stern: transom with lit cabin windows and lanterns.
  const st = side(STERN, 1), sp = side(STERN, -1);
  const ctr: V3 = [STERN, 0, 3];
  for (const [u, v] of [[st.gun, st.b1], [st.b1, st.b2], [st.b2, st.b3], [st.b3, st.wl], [st.wl, st.bilge], [st.bilge, st.keel], [sp.keel, sp.bilge], [sp.bilge, sp.wl], [sp.wl, sp.b3], [sp.b3, sp.b2], [sp.b2, sp.b1], [sp.b1, sp.gun]] as [V3, V3][]) m.tri(ctr, u, v, u[2] > 3.4 ? WOOD_DARK : HULL);
  m.tri(ctr, st.gun, sp.gun, WOOD_DARK);
  for (const y of [-2.4, -0.8, 0.8, 2.4]) m.box(STERN - 0.05, STERN + 0.02, y - 0.5, y + 0.5, 5.4, 6.4, [2.85, 2.35, 1.5]);
  m.box(STERN - 0.08, STERN, -3.3, 3.3, 6.5, 6.7, CAP);
  for (const y of [-2.2, 2.2]) { m.tube([STERN + 0.3, y, gunZ(STERN)], [STERN - 0.4, y, gunZ(STERN) + 1.2], 0.06, 0.06, 4, WALE); m.box(STERN - 0.65, STERN - 0.15, y - 0.25, y + 0.25, gunZ(STERN) + 1.2, gunZ(STERN) + 1.9, [2.95, 2.5, 1.6]); }
  // Gun ports for the deck cannons and painted lower-deck ports.
  for (const c of CANNONS) {
    const w = halfBeam(c.x) * 0.985 + 0.02;
    m.quad([c.x - 0.45, c.side * w, 4.35], [c.x + 0.45, c.side * w, 4.35], [c.x + 0.45, c.side * w, 5.05], [c.x - 0.45, c.side * w, 5.05], [0.02, 0.02, 0.02]);
  }
  for (let x = -12; x <= 11; x += 3) for (const s of [1, -1]) {
    const w = halfBeam(x) * 0.995 + 0.02;
    m.quad([x - 0.35, s * w, 2.25], [x + 0.35, s * w, 2.25], [x + 0.35, s * w, 2.85], [x - 0.35, s * w, 2.85], [0.03, 0.03, 0.025]);
  }
  // Bowsprit and stays.
  m.tube([16.5, 0, 5.6], [27, 0, 9.8], 0.32, 0.18, 8, MAST);
  m.tube([27, 0, 9.8], [9.5, 0, 33], 0.05, 0.05, 4, ROPE);
  m.tube([9.5, 0, 36], [0, 0, 40.5], 0.05, 0.05, 4, ROPE);
  m.tube([0, 0, 38], [-15.5, 0, 30], 0.05, 0.05, 4, ROPE);
  // Masts, yards, crow's nest, shrouds.
  for (const mt of MASTS) {
    m.tube([mt.x, 0, mt.base - 0.1], [mt.x, 0, mt.top], mt.r, mt.r * 0.5, 10, MAST);
    for (const y of mt.yards) m.tube([mt.x + 0.5, -y.w / 2 - 0.4, y.z], [mt.x + 0.5, y.w / 2 + 0.4, y.z], 0.22, 0.22, 6, MAST);
    const nest = mt.yards[0].z + 3;
    m.cyl(mt.x, 0, nest - 0.35, nest, 1.6, 1.6, 12, WOOD_DARK);
    for (const s of [1, -1]) for (let k = -1; k <= 1; k++) {
      const bx = mt.x + k * 1.1;
      m.tube([bx, s * (halfBeam(bx) * 0.95), gunZ(bx)], [mt.x, s * 1.3, nest - 0.3], 0.045, 0.045, 4, ROPE);
      m.tube([mt.x, s * 1.4, nest], [mt.x, s * 0.2, mt.top * 0.94], 0.04, 0.04, 4, ROPE);
    }
    for (let z = gunZ(mt.x) + 1.4; z < nest - 1; z += 1.4) for (const s of [1, -1]) {
      const t = (z - gunZ(mt.x)) / (nest - gunZ(mt.x));
      const y = s * (halfBeam(mt.x) * 0.95 * (1 - t) + 1.3 * t);
      m.tube([mt.x - 1.1 * (1 - t), y, z], [mt.x + 1.1 * (1 - t), y, z], 0.03, 0.03, 3, ROPE);
    }
  }
  // Jolly Roger.
  m.quad([-0.3, 0, 42.4], [-4.3, 0, 42.4], [-4.3, 0, 39.9], [-0.3, 0, 39.9], [0.02, 0.02, 0.02]);
  m.box(-2.6, -2.0, -0.02, 0.02, 40.9, 41.6, [1.0, 1.0, 0.95]);
  // Helm pedestal, capstan, hatches, barrels, harpoon post.
  m.box(HELM.x - 0.2, HELM.x + 0.2, -0.22, 0.22, QUARTER_DECK, HELM.z - 0.25, WOOD_DARK);
  m.cyl(-7, 0, MAIN_DECK, MAIN_DECK + 1.1, 0.7, 0.55, 10, CAP);
  m.box(3.2, 5.6, -1.3, 1.3, MAIN_DECK, MAIN_DECK + 0.35, [0.2, 0.14, 0.08], [0.12, 0.08, 0.05]);
  m.box(-4, -2.4, -1.0, 1.0, MAIN_DECK, MAIN_DECK + 0.35, [0.2, 0.14, 0.08], [0.12, 0.08, 0.05]);
  for (const [bx, by] of [[-14.5, 2.8], [-15.2, 2.2], [11.2, -2.6], [-8, 3.6], [-8.8, 3.4]]) m.cyl(bx, by, deckZ(bx, by), deckZ(bx, by) + 1.1, 0.42, 0.42, 9, [0.33, 0.22, 0.12]);
  m.cyl(HARPOON.x, 0, FORE_DECK, HARPOON.z - 0.15, 0.35, 0.28, 8, WOOD_DARK);
  return m.data;
}

// Unit sail hanging down from its yard: width 1 along y, height 1 down, bellied forward.
export function buildSailMesh(): number[] {
  const m = new MeshBuilder();
  const NU = 6, NV = 5, col: V3 = [0.74, 0.7, 0.6];
  const p = (i: number, j: number): V3 => {
    const u = i / NU, v = j / NV;
    return [0.1 * Math.sin(Math.PI * u) * Math.sin(Math.PI * Math.min(v * 1.2, 1)), u - 0.5, -v];
  };
  for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) m.quad(p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1), j % 2 ? col : [0.7, 0.66, 0.57]);
  return m.data;
}

// Cannon in its own frame: carriage on the deck, barrel along +x.
export function buildCannonMesh(): number[] {
  const m = new MeshBuilder();
  const IRON: V3 = [0.09, 0.09, 0.1];
  m.box(-1.0, 0.5, -0.45, 0.45, 0.15, 0.55, [0.3, 0.19, 0.1]);
  for (const x of [-0.8, 0.3]) for (const y of [-0.5, 0.42]) m.box(x - 0.18, x + 0.18, y, y + 0.08, 0, 0.36, [0.18, 0.11, 0.06]);
  m.tube([-1.15, 0, 0.75], [1.7, 0, 0.75], 0.27, 0.19, 10, IRON, true);
  m.tube([1.55, 0, 0.75], [1.75, 0, 0.75], 0.24, 0.24, 10, IRON, true);
  m.sphere([-1.2, 0, 0.75], 0.12, 0.12, 0.12, 6, IRON);
  return m.data;
}

// Bow harpoon (a ballista) in its pivot frame, aiming along +x.
export function buildHarpoonMesh(): number[] {
  const m = new MeshBuilder();
  const W: V3 = [0.32, 0.2, 0.11], IRON: V3 = [0.12, 0.12, 0.13];
  m.box(-0.4, 0.4, -0.35, 0.35, -0.15, 0.1, IRON);
  m.box(-1.6, 1.4, -0.14, 0.14, 0.05, 0.3, W);
  for (const s of [1, -1]) { m.tube([1.2, 0, 0.2], [0.4, s * 1.6, 0.2], 0.08, 0.05, 5, W); m.tube([0.4, s * 1.6, 0.2], [-0.9, 0, 0.32], 0.015, 0.015, 3, [0.8, 0.78, 0.7]); }
  m.box(-1.9, -1.4, -0.3, 0.3, -0.05, 0.25, IRON);
  return m.data;
}
export function buildBoltMesh(): number[] {
  const m = new MeshBuilder();
  m.tube([-1.0, 0, 0], [1.6, 0, 0], 0.05, 0.05, 5, [0.35, 0.25, 0.14]);
  m.tube([1.6, 0, 0], [2.2, 0, 0], 0.13, 0.0, 6, [0.5, 0.5, 0.52]);
  for (const s of [1, -1]) m.tri([1.75, s * 0.24, 0], [1.6, 0, 0], [2.0, 0, 0], [0.5, 0.5, 0.52]);
  return m.data;
}
export function buildBallMesh(): number[] {
  const m = new MeshBuilder();
  m.sphere([0, 0, 0], 0.22, 0.22, 0.22, 8, [0.05, 0.05, 0.05]);
  return m.data;
}

export function buildWheelMesh(): number[] {
  const m = new MeshBuilder();
  const R = 0.3, WOOD: V3 = [0.35, 0.22, 0.12], BRASS: V3 = [0.55, 0.45, 0.2];
  const seg = 18;
  for (let i = 0; i < seg; i++) {
    const a0 = i / seg * Math.PI * 2, a1 = (i + 1) / seg * Math.PI * 2;
    const p = (a: number, r: number, x: number): V3 => [x, Math.cos(a) * r, Math.sin(a) * r];
    m.quad(p(a0, R, -0.02), p(a1, R, -0.02), p(a1, R, 0.02), p(a0, R, 0.02), WOOD);
    m.quad(p(a0, R - 0.04, -0.02), p(a1, R - 0.04, -0.02), p(a1, R, -0.02), p(a0, R, -0.02), WOOD);
    m.quad(p(a0, R - 0.04, 0.02), p(a1, R - 0.04, 0.02), p(a1, R, 0.02), p(a0, R, 0.02), WOOD);
  }
  for (let i = 0; i < 6; i++) {
    const a = i / 6 * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a), w = 0.015;
    m.quad([0, -sa * w, ca * w], [0, ca * (R + 0.08) - sa * w, sa * (R + 0.08) + ca * w], [0, ca * (R + 0.08) + sa * w, sa * (R + 0.08) - ca * w], [0, sa * w, -ca * w], i === 0 ? BRASS : WOOD);
  }
  m.box(-0.06, 0.04, -0.05, 0.05, -0.05, 0.05, BRASS);
  return m.data;
}

// ---- Physics ----
export function shipAxes(psi: number, th: number, ph: number) {
  const cp = Math.cos(psi), sp = Math.sin(psi), ct = Math.cos(th), st = Math.sin(th), cf = Math.cos(ph), sf = Math.sin(ph);
  const X: V3 = [cp * ct, sp * ct, st];
  const Y: V3 = [-cp * st * sf - sp * cf, -sp * st * sf + cp * cf, ct * sf];
  const Z: V3 = [-cp * st * cf + sp * sf, -sp * st * cf - cp * sf, ct * cf];
  return { X, Y, Z };
}

export interface ShipEvent { type: 'slam' | 'crash' | 'green'; i: number; }

export class Ship {
  x = 0; y = -300; psi = -Math.PI / 2; vx = 0; vy = 0; r = 0;
  z = 0; vz = 0; th = 0; thv = 0; ph = 0; phv = 0;
  rudder = 0; sail = 0; sailSet = 0.08; hull = 100;
  u = 0; imm = 0; waterZ = 0;
  events: ShipEvent[] = [];
  private slamCd = 0; private crashCd = 0; private greenCd = 0;
  private sm = [newSample(), newSample(), newSample(), newSample(), newSample()];

  place(x: number, y: number, psi: number) {
    this.x = x; this.y = y; this.psi = psi; this.vx = this.vy = this.r = this.vz = this.thv = this.phv = 0;
    this.z = sampleWater(x, y, this.sm[4]).z; this.th = this.ph = 0;
  }
  local(lx: number, ly: number, lz: number): V3 {
    const { X, Y, Z } = shipAxes(this.psi, this.th, this.ph);
    return [this.x + X[0] * lx + Y[0] * ly + Z[0] * lz, this.y + X[1] * lx + Y[1] * ly + Z[1] * lz, this.z + X[2] * lx + Y[2] * ly + Z[2] * lz];
  }
  impulse(ix: number, iy: number, roll: number) { this.vx += ix; this.vy += iy; this.phv += roll; }
  // Pull toward a world point when a rope of length len is stretched (harpoon or gangplank lashing).
  pullToward(px: number, py: number, fromX: number, fromY: number, len: number, k: number, dt: number) {
    const dx = px - fromX, dy = py - fromY, d = Math.hypot(dx, dy);
    if (d <= len || d < 1e-3) return;
    const nx = dx / d, ny = dy / d, ex = d - len;
    this.vx += nx * ex * k * dt; this.vy += ny * ex * k * dt;
    const away = -(this.vx * nx + this.vy * ny);
    if (away > 0) { this.vx += nx * away * Math.min(1, 3 * dt); this.vy += ny * away * Math.min(1, 3 * dt); }
  }

  step(dt: number, windX: number, windY: number) {
    const c = Math.cos(this.psi), s = Math.sin(this.psi);
    const fx = c, fy = s, lx = -s, ly = c;
    const pts = [[16, 0], [-16, 0], [0, 4.6], [0, -4.6], [0, 0]];
    for (let i = 0; i < 5; i++) sampleWater(this.x + fx * pts[i][0] + lx * pts[i][1], this.y + fy * pts[i][0] + ly * pts[i][1], this.sm[i]);
    const [b, st, pt, sb, ce] = this.sm;
    const zT = (b.z + st.z + pt.z + sb.z + 2 * ce.z) / 6;
    this.waterZ = ce.z;
    const thT = Math.atan2(b.z - st.z, 32), phT = Math.atan2(pt.z - sb.z, 9.2);
    const wvx = (b.vx + st.vx + pt.vx + sb.vx + ce.vx) / 5 + windX * 0.03;
    const wvy = (b.vy + st.vy + pt.vy + sb.vy + ce.vy) / 5 + windY * 0.03;
    const along = (b.z - st.z) / 32, across = (pt.z - sb.z) / 9.2;
    const gx = fx * along + lx * across, gy = fy * along + ly * across;

    // Sails unfurl slowly toward the set; they drive best with the wind behind.
    this.sailSet += clamp(Math.max(this.sail, 0.08) - this.sailSet, -0.25 * dt, 0.25 * dt);
    const windAng = this.psi - WIND_DIR;
    const eff = 0.35 + 0.65 * (0.5 + 0.5 * Math.cos(windAng));
    const rvx = this.vx - wvx, rvy = this.vy - wvy;
    const u = rvx * fx + rvy * fy, v = rvx * lx + rvy * ly;
    this.u = u;
    const thrust = Math.max(this.sailSet - 0.08, 0) / 0.92 * eff * 1.0;
    const fu = thrust - 0.02 * u - 0.012 * u * Math.abs(u);
    const fv = -1.0 * v - 0.2 * v * Math.abs(v);
    const ax = fx * fu + lx * fv - 9.81 * 0.8 * gx + windX * 0.002;
    const ay = fy * fu + ly * fv - 9.81 * 0.8 * gy + windY * 0.002;
    this.vx += ax * dt; this.vy += ay * dt;

    const auth = 0.012 + 0.06 * clamp(Math.abs(u) / 6, 0, 1);
    const rT = this.rudder * auth * (u < -0.3 ? -1 : 1);
    this.r += ((rT - this.r) * 0.7 - 0.02 * (gx * lx + gy * ly) * clamp(u / 4, -1, 1)) * dt;
    this.psi += this.r * dt;
    this.x += this.vx * dt; this.y += this.vy * dt;

    this.vz += ((zT + 0.4 - this.z) * 4.6 - 3.0 * this.vz) * dt; this.z += this.vz * dt;
    this.thv += ((thT - this.th) * 4.0 - 2.6 * this.thv) * dt; this.th = clamp(this.th + this.thv * dt, -0.6, 0.6);
    const heel = clamp(this.r * u * 0.25, -0.12, 0.12) + this.sailSet * 0.07 * Math.sin(windAng);
    this.phv += ((phT + heel - this.ph) * 3.0 - 1.3 * this.phv) * dt; this.ph = clamp(this.ph + this.phv * dt, -0.6, 0.6);

    const bowZ = this.z + 16 * Math.sin(this.th), bowVz = this.vz + 16 * this.thv * Math.cos(this.th);
    this.imm = b.z - bowZ;
    const rel = b.vz - bowVz;
    this.slamCd -= dt; this.crashCd -= dt; this.greenCd -= dt;
    if (this.imm > -1.5 && rel > 2.0 && this.slamCd <= 0) {
      const i = clamp((rel - 2.0) / 4 + this.imm * 0.1, 0.15, 1);
      this.events.push({ type: 'slam', i });
      this.slamCd = 0.9;
      this.thv += i * 0.12; this.vz += i * 0.5;
      this.vx -= fx * u * 0.06 * i; this.vy -= fy * u * 0.06 * i;
    }
    if (this.imm > 1.2 && this.greenCd <= 0) { this.events.push({ type: 'green', i: clamp((this.imm - 1.2) / 3, 0.2, 1) }); this.greenCd = 1.2; }
    this.collide(fx, fy, lx, ly);
    if (this.hull > 0 && this.hull < 100) this.hull = Math.min(100, this.hull + dt * 0.4);
  }

  private collide(fx: number, fy: number, lx: number, ly: number) {
    let impact = 0;
    for (const [px, py] of [[17, 0], [8, 4.5], [8, -4.5], [-8, 4.8], [-8, -4.8], [-17, 3.5], [-17, -3.5]]) {
      const wx = this.x + fx * px + lx * py, wy = this.y + fy * px + ly * py;
      const H = terrainHeight(wx, wy);
      if (H <= -3.8) continue;
      const e = 2;
      const gx = (terrainHeight(wx + e, wy) - terrainHeight(wx - e, wy)) / (2 * e);
      const gy = (terrainHeight(wx, wy + e) - terrainHeight(wx, wy - e)) / (2 * e);
      const gl = Math.hypot(gx, gy);
      if (gl < 1e-4) continue;
      const nx = gx / gl, ny = gy / gl;
      const pen = Math.min((H + 3.8) / Math.max(gl, 0.3), 2);
      this.x -= nx * pen; this.y -= ny * pen;
      const vn = this.vx * nx + this.vy * ny;
      if (vn > 0) { this.vx -= nx * vn * 1.5; this.vy -= ny * vn * 1.5; impact = Math.max(impact, vn); }
    }
    for (const p of PILLARS) for (const hx of [14, 7, 0, -7, -14]) {
      const cx = this.x + fx * hx, cy = this.y + fy * hx;
      const dx = cx - p.x, dy = cy - p.y, d = Math.hypot(dx, dy), minD = p.r + 4.5;
      if (d >= minD || d < 1e-3) continue;
      const nx = dx / d, ny = dy / d;
      this.x += nx * (minD - d); this.y += ny * (minD - d);
      const vn = -(this.vx * nx + this.vy * ny);
      if (vn > 0) { this.vx += nx * vn * 1.5; this.vy += ny * vn * 1.5; impact = Math.max(impact, vn); }
    }
    if (impact > 0.3) {
      this.vx *= 0.7; this.vy *= 0.7; this.r *= 0.4;
      if (this.crashCd <= 0) { this.events.push({ type: 'crash', i: clamp(impact / 4, 0.2, 1) }); this.crashCd = 0.8; this.hull = Math.max(0, this.hull - impact * 3); }
    }
  }

  pack(): number[] {
    return [this.x, this.y, this.psi, this.vx, this.vy, this.r, this.z, this.vz, this.th, this.thv, this.ph, this.phv, this.rudder, this.sail, this.sailSet, this.hull];
  }
  unpack(a: number[]) {
    [this.x, this.y, this.psi, this.vx, this.vy, this.r, this.z, this.vz, this.th, this.thv, this.ph, this.phv, this.rudder, this.sail, this.sailSet, this.hull] = a;
  }
}
