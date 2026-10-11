// A 14 m trawler. Local frame: x forward, y to port (left), z up, z = 0 at the waterline.
import { MeshBuilder, V3 } from './mesh';
import { sampleWater, newSample } from './waves';
import { terrainHeight, PILLARS } from './island';
import { clamp } from './world';

export const LEN = 14, HALF_LEN = 7;
// Wheelhouse interior (camera lives here) and the eye position.
export const CAB = { x0: -3.6, x1: 0.3, y: 1.75, floor: 1.5, sill: 2.6, roof: 4.25 };
export const EYE: V3 = [-1.25, 0, 3.3];
export const WHEEL_POS: V3 = [-0.45, 0, 2.45];

const halfBeam = (t: number) => t < 0.55 ? 2.3 * (0.82 + 0.18 * Math.sin(Math.PI / 2 * t / 0.55)) : 2.3 * Math.pow(Math.cos((t - 0.55) / 0.45 * Math.PI / 2), 0.8);
const sheer = (t: number) => 1.5 + 0.9 * Math.pow(t, 2.5);
const keel = (t: number) => (t < 0.7 ? -1.6 : -1.6 * (1 - Math.pow((t - 0.7) / 0.3, 1.5) * 0.85)) * (t < 0.1 ? 0.8 + 2 * t : 1);

export function buildBoatMesh(): number[] {
  const m = new MeshBuilder();
  const NAVY: V3 = [0.07, 0.1, 0.17], WHITE: V3 = [0.78, 0.78, 0.75], RED: V3 = [0.42, 0.09, 0.07];
  const DECK: V3 = [0.3, 0.28, 0.24], CAP: V3 = [0.32, 0.22, 0.13], INNER: V3 = [0.62, 0.62, 0.6];
  const N = 22;
  type Sec = { x: number; w: number; s: number; k: number; d: number };
  const secs: Sec[] = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    secs.push({ x: -HALF_LEN + t * LEN, w: Math.max(halfBeam(t), 0.02), s: sheer(t), k: keel(t), d: sheer(t) - 0.6 });
  }
  const side = (sec: Sec, sgn: number) => ({
    gun: [sec.x, sgn * sec.w, sec.s] as V3,
    boot: [sec.x, sgn * sec.w, 0.35] as V3,
    wl: [sec.x, sgn * sec.w * 0.99, 0.12] as V3,
    chine: [sec.x, sgn * sec.w * 0.93, -0.3] as V3,
    bilge: [sec.x, sgn * sec.w * 0.55, sec.k * 0.78] as V3,
    keel: [sec.x, 0, sec.k] as V3,
    capIn: [sec.x, sgn * Math.max(sec.w - 0.15, 0), sec.s] as V3,
    deckIn: [sec.x, sgn * Math.max(sec.w - 0.15, 0), sec.d] as V3,
  });
  for (let i = 0; i < N; i++) {
    for (const sgn of [1, -1]) {
      const a = side(secs[i], sgn), b = side(secs[i + 1], sgn);
      m.quad(a.gun, b.gun, b.boot, a.boot, NAVY);
      m.quad(a.boot, b.boot, b.wl, a.wl, WHITE);
      m.quad(a.wl, b.wl, b.chine, a.chine, RED);
      m.quad(a.chine, b.chine, b.bilge, a.bilge, RED);
      m.quad(a.bilge, b.bilge, b.keel, a.keel, RED);
      m.quad(a.gun, b.gun, b.capIn, a.capIn, CAP);
      m.quad(a.capIn, b.capIn, b.deckIn, a.deckIn, INNER);
    }
    const a = secs[i], b = secs[i + 1];
    const wa = Math.max(a.w - 0.15, 0), wb = Math.max(b.w - 0.15, 0);
    m.quad([a.x, -wa, a.d], [b.x, -wb, b.d], [b.x, wb, b.d], [a.x, wa, a.d], DECK);
  }
  // Transom.
  const s0 = secs[0], p = side(s0, 1), q = side(s0, -1);
  const ctr: V3 = [s0.x, 0, 0.4];
  for (const [u, v] of [[p.gun, p.boot], [p.boot, p.wl], [p.wl, p.chine], [p.chine, p.bilge], [p.bilge, p.keel], [q.keel, q.bilge], [q.bilge, q.chine], [q.chine, q.wl], [q.wl, q.boot], [q.boot, q.gun]] as [V3, V3][]) {
    m.tri(ctr, u, v, u[2] > 0.35 ? NAVY : u[2] > 0.1 ? WHITE : RED);
  }
  m.tri(ctr, p.gun, q.gun, NAVY);

  // Wheelhouse: lower walls, corner posts and mullions, open windows, roof with a visor.
  const { x0, x1, y, floor, sill, roof } = CAB;
  const WALL: V3 = [0.8, 0.8, 0.77], POST: V3 = [0.13, 0.15, 0.19], DARK: V3 = [0.07, 0.07, 0.08];
  m.box(x0, x1, -y, -y + 0.08, 0.95, sill, WALL); m.box(x0, x1, y - 0.08, y, 0.95, sill, WALL);
  m.box(x1 - 0.08, x1, -y, y, 0.95, sill, WALL);
  m.box(x0, x0 + 0.08, -y, y, 0.95, roof, WALL);
  m.box(x0, x1, -y, y, floor - 0.06, floor, [0.22, 0.17, 0.12]);
  for (const py of [-y, -0.72, 0.62, y - 0.1]) m.box(x1 - 0.1, x1, py, py + 0.1, sill, roof, POST);
  for (const px of [x0, -1.65, x1 - 0.1]) for (const py of [-y, y - 0.08]) m.box(px, px + 0.1, py, py + 0.08, sill, roof, POST);
  m.box(x0 - 0.25, x1 + 0.35, -y - 0.15, y + 0.15, roof, roof + 0.16, [0.84, 0.84, 0.82]);
  m.box(x0, x1, -y, y, roof - 0.04, roof, [0.2, 0.2, 0.21]);
  m.box(x1 - 0.1, x1 + 0.35, -y - 0.15, y + 0.15, roof - 0.1, roof, DARK);
  // Dashboard, gauges, compass, lever.
  m.box(-0.05, x1 - 0.08, -1.6, 1.6, floor, sill - 0.05, DARK, [0.13, 0.13, 0.14]);
  for (const gy of [-1.15, -0.85, 0.85, 1.15]) m.box(-0.07, -0.05, gy - 0.06, gy + 0.06, sill - 0.3, sill - 0.2, gy < 0 ? [2.08, 2.3, 2.12] : [2.35, 2.2, 2.05]);
  m.box(-0.3, -0.08, -0.11, 0.11, sill - 0.05, sill + 0.08, [0.05, 0.05, 0.05]);
  m.box(-0.27, -0.11, -0.08, 0.08, sill + 0.08, sill + 0.09, [2.25, 2.2, 2.05]);
  m.box(-0.35, -0.25, 0.55, 0.62, sill - 0.05, sill + 0.25, [0.6, 0.6, 0.62]);
  // Mast with lights, exhaust stack, running lights, life ring.
  m.box(-0.1, 0.05, -0.06, 0.06, roof, roof + 2.4, [0.6, 0.62, 0.66]);
  m.box(-0.1, 0.05, -0.7, 0.7, roof + 1.7, roof + 1.8, [0.6, 0.62, 0.66]);
  m.box(-0.13, 0.08, -0.08, 0.08, roof + 2.4, roof + 2.6, [2.95, 2.95, 2.9]);
  m.box(x1 - 0.1, x1 + 0.05, y, y + 0.08, sill - 0.3, sill - 0.1, [2.9, 2.1, 2.1]);
  m.box(x1 - 0.1, x1 + 0.05, -y - 0.08, -y, sill - 0.3, sill - 0.1, [2.1, 2.9, 2.2]);
  m.box(-4.4, -3.9, 1.0, 1.5, 0.95, roof + 0.7, [0.75, 0.75, 0.73]);
  m.box(-4.42, -3.88, 0.98, 1.52, roof + 0.5, roof + 0.75, [0.05, 0.05, 0.05]);
  for (const [a, b, c, d] of [[-2.9, -2.1, 0, 0.12], [-2.9, -2.1, 0.68, 0.8], [-2.9, -2.78, 0, 0.8], [-2.22, -2.1, 0, 0.8]]) m.box(a, b, y + 0.08, y + 0.14, 1.4 + c, 1.4 + d, [0.95, 0.4, 0.06]);
  // Foredeck: hatch, windlass, bollards, bow pulpit.
  m.box(1.4, 3.0, -0.9, 0.9, 1.0, 1.55, [0.12, 0.2, 0.17], [0.15, 0.24, 0.2]);
  m.box(5.0, 5.7, -0.45, 0.45, 1.65, 2.05, [0.2, 0.2, 0.22]);
  for (const [bx, by] of [[4.2, 1.2], [4.2, -1.2], [-5.5, 1.4], [-5.5, -1.4]]) m.box(bx - 0.12, bx + 0.12, by - 0.12, by + 0.12, sheer((bx + 7) / 14) - 0.6, sheer((bx + 7) / 14) - 0.2, [0.15, 0.15, 0.15]);
  m.box(6.4, 7.3, -0.04, 0.04, 2.35, 2.42, [0.6, 0.62, 0.66]);
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

// Rotation for heading psi, pitch theta (nose up +), roll phi (starboard down +): Rz(psi) Ry(-theta) Rx(phi).
export function boatAxes(psi: number, th: number, ph: number) {
  const cp = Math.cos(psi), sp = Math.sin(psi), ct = Math.cos(th), st = Math.sin(th), cf = Math.cos(ph), sf = Math.sin(ph);
  const X: V3 = [cp * ct, sp * ct, st];
  const Y: V3 = [-cp * st * sf - sp * cf, -sp * st * sf + cp * cf, ct * sf];
  const Z: V3 = [-cp * st * cf + sp * sf, -sp * st * cf - cp * sf, ct * cf];
  return { X, Y, Z };
}

export interface BoatEvent { type: 'slam' | 'crash' | 'green'; i: number; }

export class Boat {
  x = 0; y = -360; psi = -Math.PI / 2; vx = 0; vy = 0; r = 0;
  z = 0; vz = 0; th = 0; thv = 0; ph = 0; phv = 0;
  lever = 0; rudder = 0; rpm = 0;
  u = 0; imm = 0; wvx = 0; wvy = 0; waterZ = 0;
  events: BoatEvent[] = [];
  private slamCd = 0; private crashCd = 0; private greenCd = 0;
  private sm = [newSample(), newSample(), newSample(), newSample(), newSample()];

  local(lx: number, ly: number, lz: number): V3 {
    const { X, Y, Z } = boatAxes(this.psi, this.th, this.ph);
    return [this.x + X[0] * lx + Y[0] * ly + Z[0] * lz, this.y + X[1] * lx + Y[1] * ly + Z[1] * lz, this.z + X[2] * lx + Y[2] * ly + Z[2] * lz];
  }

  step(dt: number, windX: number, windY: number) {
    const c = Math.cos(this.psi), s = Math.sin(this.psi);
    const fx = c, fy = s, lx = -s, ly = c;
    const pts = [[6.2, 0], [-6.2, 0], [0, 2.1], [0, -2.1], [0, 0]];
    for (let i = 0; i < 5; i++) sampleWater(this.x + fx * pts[i][0] + lx * pts[i][1], this.y + fy * pts[i][0] + ly * pts[i][1], this.sm[i]);
    const [b, st, pt, sb, ce] = this.sm;
    const zT = (b.z + st.z + pt.z + sb.z + 2 * ce.z) / 6;
    this.waterZ = ce.z;
    const thT = Math.atan2(b.z - st.z, 12.4), phT = Math.atan2(pt.z - sb.z, 4.2);
    // Orbital currents plus a steady downwind drift (Stokes drift) that grows with the storm.
    this.wvx = (b.vx + st.vx + pt.vx + sb.vx + ce.vx) / 5 + windX * 0.03;
    this.wvy = (b.vy + st.vy + pt.vy + sb.vy + ce.vy) / 5 + windY * 0.03;
    const along = (b.z - st.z) / 12.4, across = (pt.z - sb.z) / 4.2;
    const gx = fx * along + lx * across, gy = fy * along + ly * across;

    // Engine spools toward the lever.
    this.rpm += (this.lever - this.rpm) * (1 - Math.exp(-dt / 0.8));
    // Hydrodynamics relative to the moving water: the boat rides the orbital currents.
    const rvx = this.vx - this.wvx, rvy = this.vy - this.wvy;
    const u = rvx * fx + rvy * fy, v = rvx * lx + rvy * ly;
    this.u = u;
    const thrust = this.rpm > 0 ? this.rpm * 1.7 : this.rpm * 1.1;
    const fu = thrust - 0.03 * u - 0.0115 * u * Math.abs(u);
    const fv = -1.6 * v - 0.25 * v * Math.abs(v);
    // Gravity along the wave face: the boat slides down slopes like on a hill.
    const ax = fx * fu + lx * fv - 9.81 * 0.8 * gx + windX * 0.004;
    const ay = fy * fu + ly * fv - 9.81 * 0.8 * gy + windY * 0.004;
    this.vx += ax * dt; this.vy += ay * dt;

    // Steering: strong at speed, some authority from prop wash when idle.
    const auth = 0.05 + 0.18 * clamp(Math.abs(u) / 7, 0, 1) + 0.06 * Math.max(this.rpm, 0);
    const rT = this.rudder * auth * (u < -0.3 ? -1 : 1);
    const waveYaw = -0.05 * (gx * lx + gy * ly) * clamp(u / 5, -1, 1);
    this.r += ((rT - this.r) * 1.3 + waveYaw) * dt;
    this.psi += this.r * dt;
    this.x += this.vx * dt; this.y += this.vy * dt;

    // Buoyancy: heave, pitch and roll chase the sea with springy lag; heel outward in turns.
    this.vz += ((zT - this.z) * 5 - 3 * this.vz) * dt; this.z += this.vz * dt;
    this.thv += ((thT - this.th) * 6 - 2.6 * this.thv) * dt; this.th = clamp(this.th + this.thv * dt, -0.55, 0.55);
    const heel = clamp(this.r * u * 0.03, -0.15, 0.15);
    this.phv += ((phT + heel - this.ph) * 3.6 - 1.5 * this.phv) * dt; this.ph = clamp(this.ph + this.phv * dt, -0.6, 0.6);

    // Slamming: water rising fast into the bow.
    const bowZ = this.z + 6.2 * Math.sin(this.th), bowVz = this.vz + 6.2 * this.thv * Math.cos(this.th);
    this.imm = b.z - bowZ;
    const rel = b.vz - bowVz;
    this.slamCd -= dt; this.crashCd -= dt; this.greenCd -= dt;
    if (this.imm > -0.4 && rel > 1.6 && this.slamCd <= 0) {
      const i = clamp((rel - 1.6) / 3.5 + this.imm * 0.15, 0.12, 1);
      this.events.push({ type: 'slam', i });
      this.slamCd = 0.55;
      this.thv += i * 0.3; this.vz += i * 0.7;
      this.vx -= fx * u * 0.1 * i; this.vy -= fy * u * 0.1 * i;
    }
    if (this.imm > 1.3 && this.greenCd <= 0) { this.events.push({ type: 'green', i: clamp((this.imm - 1.3) / 1.5, 0.2, 1) }); this.greenCd = 0.8; }

    this.collide(fx, fy, lx, ly);
  }

  private collide(fx: number, fy: number, lx: number, ly: number) {
    let impact = 0;
    for (const [px, py] of [[6.6, 0], [3, 1.9], [3, -1.9], [-5, 1.8], [-5, -1.8]]) {
      const wx = this.x + fx * px + lx * py, wy = this.y + fy * px + ly * py;
      const H = terrainHeight(wx, wy);
      if (H <= -1.4) continue;
      const e = 1.5;
      const gx = (terrainHeight(wx + e, wy) - terrainHeight(wx - e, wy)) / (2 * e);
      const gy = (terrainHeight(wx, wy + e) - terrainHeight(wx, wy - e)) / (2 * e);
      const gl = Math.hypot(gx, gy);
      if (gl < 1e-4) continue;
      const nx = gx / gl, ny = gy / gl; // uphill = toward land
      const pen = Math.min((H + 1.4) / Math.max(gl, 0.3), 1.5);
      this.x -= nx * pen; this.y -= ny * pen;
      const vn = this.vx * nx + this.vy * ny;
      if (vn > 0) { this.vx -= nx * vn * 1.5; this.vy -= ny * vn * 1.5; impact = Math.max(impact, vn); }
    }
    for (const p of PILLARS) for (const hx of [5, 0, -5]) {
      const cx = this.x + fx * hx, cy = this.y + fy * hx;
      const dx = cx - p.x, dy = cy - p.y, d = Math.hypot(dx, dy), minD = p.r + 2.3;
      if (d >= minD || d < 1e-3) continue;
      const nx = dx / d, ny = dy / d; // away from the rock
      this.x += nx * (minD - d); this.y += ny * (minD - d);
      const vn = -(this.vx * nx + this.vy * ny);
      if (vn > 0) { this.vx += nx * vn * 1.5; this.vy += ny * vn * 1.5; impact = Math.max(impact, vn); }
    }
    if (impact > 0.4) {
      this.vx *= 0.75; this.vy *= 0.75; this.r *= 0.5;
      if (this.crashCd <= 0) { this.events.push({ type: 'crash', i: clamp(impact / 6, 0.2, 1) }); this.crashCd = 0.6; }
    }
  }
}
