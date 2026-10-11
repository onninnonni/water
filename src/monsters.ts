// Sea serpents: host-simulated, interpolated on guests. Each serpent is a head followed by
// body segments laid along the trail of the head, arching in and out of the water.
import { MeshBuilder, V3 } from './mesh';
import { sampleWater, newSample } from './waves';
import { clamp } from './world';

export const SEGS = 14, SEG_GAP = 2.6;
export const segRadius = (i: number) => (i === 0 ? 2.3 : 2.0 * (1 - (i / SEGS) * 0.72));
export enum MS { Approach, Circle, Lunge, Retreat, Dying }

export interface Monster {
  id: number; x: number; y: number; hd: number; sp: number; phase: number; hp: number; st: MS; t: number;
  rear: number; dying: number; hurt: number; trail: number[]; segs: V3[]; orbit: number; dir: number;
}

export function buildSegmentMesh(): number[] {
  const m = new MeshBuilder();
  m.sphere([0, 0, 0], 1.05, 1, 0.95, 10, (z) => (z > 0.2 ? [0.05, 0.15, 0.13] : z > -0.3 ? [0.1, 0.22, 0.17] : [0.36, 0.38, 0.26]));
  m.tri([-0.6, 0, 0.85], [0.5, 0, 0.9], [-0.3, 0, 1.75], [0.12, 0.08, 0.1]);
  return m.data;
}
export function buildHeadMesh(): number[] {
  const m = new MeshBuilder();
  const TOP: V3 = [0.05, 0.15, 0.13], BELLY: V3 = [0.36, 0.38, 0.26];
  m.sphere([0.6, 0, 0.15], 2.3, 1.25, 1.0, 12, (z) => (z > -0.1 ? TOP : BELLY));
  m.sphere([1.4, 0, -0.75], 1.7, 0.95, 0.42, 10, BELLY);
  for (const s of [1, -1]) {
    m.sphere([1.9, s * 0.95, 0.55], 0.22, 0.22, 0.2, 6, [2.95, 2.55, 0.35]);
    m.tube([-0.3, s * 0.7, 0.8], [-2.2, s * 1.3, 2.0], 0.28, 0.04, 6, [0.55, 0.5, 0.38]);
    for (let k = 0; k < 4; k++) { const x = 1.4 + k * 0.55; m.tri([x, s * 0.85, -0.3], [x + 0.25, s * 0.85, -0.3], [x + 0.12, s * 0.8, -0.75], [0.95, 0.93, 0.85]); }
  }
  m.tri([-1.2, 0, 0.9], [0.6, 0, 1.1], [-0.8, 0, 2.3], [0.12, 0.08, 0.1]);
  return m.data;
}

const tmp = newSample();
export function layoutSegments(m: Monster, time: number) {
  const segs: V3[] = [];
  // Walk the trail (newest first) to place segments every SEG_GAP metres.
  let ax = m.x, ay = m.y, acc = 0, ti = 0;
  for (let i = 0; i < SEGS; i++) {
    const want = i * SEG_GAP;
    while (acc < want && ti < m.trail.length) {
      const bx = m.trail[ti], by = m.trail[ti + 1], d = Math.hypot(bx - ax, by - ay);
      if (acc + d >= want) { const f = (want - acc) / Math.max(d, 1e-6); ax += (bx - ax) * f; ay += (by - ay) * f; acc = want; break; }
      acc += d; ax = bx; ay = by; ti += 2;
    }
    if (acc < want) { ax -= Math.cos(m.hd) * (want - acc); ay -= Math.sin(m.hd) * (want - acc); acc = want; }
    const w = sampleWater(ax, ay, tmp).z;
    const lift = m.rear * 7 * Math.max(0, 1 - i / 4);
    const und = 2.2 * Math.sin(m.phase - i * 0.55) * (i === 0 ? 0.3 : 1);
    segs.push([ax, ay, w - 0.5 + und + lift + (i === 0 ? 1.0 : 0) - m.dying * 9 + 0.15 * Math.sin(time * 3 + i)]);
  }
  m.segs = segs;
}

export function pushTrail(m: Monster) {
  const lx = m.trail[0] ?? m.x - Math.cos(m.hd), ly = m.trail[1] ?? m.y - Math.sin(m.hd);
  if (Math.hypot(m.x - lx, m.y - ly) > 1.0) {
    m.trail.unshift(m.x, m.y);
    if (m.trail.length > (SEGS * SEG_GAP + 6) * 2) m.trail.length = Math.floor((SEGS * SEG_GAP + 6) * 2);
  }
}

export interface ShipRef { x: number; y: number; psi: number; vx: number; vy: number; }
export class MonsterSystem {
  list: Monster[] = [];
  private nextId = 1; private spawnT = 8;

  update(dt: number, ship: ShipRef, storm: number, time: number, onRam: (m: Monster, side: number) => void) {
    const max = storm < 0.12 ? 0 : 1 + Math.floor(storm * 2.6);
    this.spawnT -= dt;
    if (this.list.filter((m) => m.st !== MS.Dying).length < max && this.spawnT <= 0) {
      const a = Math.random() * Math.PI * 2, d = 260 + Math.random() * 80;
      const x = ship.x + Math.cos(a) * d, y = ship.y + Math.sin(a) * d;
      this.list.push({ id: this.nextId++, x, y, hd: a + Math.PI, sp: 9, phase: 0, hp: 4, st: MS.Approach, t: 0, rear: 0, dying: 0, hurt: 0, trail: [], segs: [], orbit: 45 + Math.random() * 30, dir: Math.random() < 0.5 ? 1 : -1 });
      this.spawnT = 10 + Math.random() * 12;
    }
    const c = Math.cos(ship.psi), s = Math.sin(ship.psi);
    for (const m of this.list) {
      m.t += dt; m.hurt = Math.max(0, m.hurt - dt * 2);
      const dx = m.x - ship.x, dy = m.y - ship.y, dist = Math.hypot(dx, dy);
      let tgt = m.hd, speed = 8;
      if (m.st === MS.Dying) { m.dying = Math.min(1, m.dying + dt / 3); speed = 2; }
      else if (storm < 0.08 && m.st !== MS.Retreat) { m.st = MS.Retreat; m.t = 0; }
      if (m.st === MS.Approach) {
        tgt = Math.atan2(-dy, -dx); speed = 10;
        if (dist < m.orbit + 15) { m.st = MS.Circle; m.t = 0; }
      } else if (m.st === MS.Circle) {
        const a = Math.atan2(dy, dx) + m.dir * 0.35;
        tgt = Math.atan2(ship.y + Math.sin(a) * m.orbit - m.y, ship.x + Math.cos(a) * m.orbit - m.x); speed = 8.5;
        if (m.t > 7 + (m.id % 5) * 1.5) { m.st = MS.Lunge; m.t = 0; }
      } else if (m.st === MS.Lunge) {
        const lead = 2.0;
        tgt = Math.atan2(ship.y + ship.vy * lead - m.y, ship.x + ship.vx * lead - m.x); speed = 15;
        const lx = dx * c + dy * s, ly = -dx * s + dy * c;
        if (Math.abs(lx) < 19 && Math.abs(ly) < 7) { onRam(m, Math.sign(ly) || 1); m.st = MS.Retreat; m.t = 0; }
        if (m.t > 12) { m.st = MS.Circle; m.t = 0; }
      } else if (m.st === MS.Retreat) {
        tgt = Math.atan2(dy, dx); speed = 11;
        if (m.t > 5 && storm >= 0.08) { m.st = MS.Circle; m.t = 0; }
      }
      const turn = Math.atan2(Math.sin(tgt - m.hd), Math.cos(tgt - m.hd));
      m.hd += clamp(turn, -0.9 * dt, 0.9 * dt);
      m.sp += (speed - m.sp) * (1 - Math.exp(-dt));
      m.x += Math.cos(m.hd) * m.sp * dt; m.y += Math.sin(m.hd) * m.sp * dt;
      m.phase += dt * (1.6 + m.sp * 0.12);
      const wantRear = m.st === MS.Lunge && dist < 70 ? 1 : m.st === MS.Circle ? 0.25 + 0.2 * Math.sin(m.t * 0.7) : 0;
      m.rear += (wantRear - m.rear) * (1 - Math.exp(-dt * 1.5));
      pushTrail(m);
      layoutSegments(m, time);
    }
    this.list = this.list.filter((m) => !(m.dying >= 1 || (m.st === MS.Retreat && storm < 0.08 && Math.hypot(m.x - ship.x, m.y - ship.y) > 400)));
  }

  // Returns the monster hit by a point, if any.
  hitTest(p: V3, pad: number): Monster | null {
    for (const m of this.list) {
      if (m.st === MS.Dying) continue;
      for (let i = 0; i < m.segs.length; i++) {
        const q = m.segs[i], r = segRadius(i) + pad;
        if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < r * r) return m;
      }
    }
    return null;
  }
  damage(m: Monster, dmg: number): boolean {
    m.hp -= dmg; m.hurt = 1;
    if (m.hp <= 0 && m.st !== MS.Dying) { m.st = MS.Dying; m.t = 0; return true; }
    if (m.st === MS.Lunge || m.st === MS.Circle) { m.st = MS.Retreat; m.t = 0; }
    return false;
  }
  pack(): number[][] { return this.list.map((m) => [m.id, +m.x.toFixed(2), +m.y.toFixed(2), +m.hd.toFixed(3), +m.phase.toFixed(3), m.hp, m.st, +m.rear.toFixed(3), +m.dying.toFixed(3), +m.hurt.toFixed(2)]); }
  // Guest side: apply a snapshot, keeping local trails.
  apply(rows: number[][], time: number) {
    const keep = new Map(this.list.map((m) => [m.id, m]));
    this.list = rows.map((r) => {
      const m = keep.get(r[0]) ?? { id: r[0], x: r[1], y: r[2], hd: r[3], sp: 0, phase: 0, hp: 4, st: MS.Approach, t: 0, rear: 0, dying: 0, hurt: 0, trail: [], segs: [], orbit: 60, dir: 1 };
      [m.x, m.y, m.hd, m.phase, m.hp, m.st, m.rear, m.dying, m.hurt] = r.slice(1) as [number, number, number, number, number, MS, number, number, number];
      pushTrail(m); layoutSegments(m, time);
      return m;
    });
  }
}
