// Sea serpents and the Leviathan: simulated by the world host, smoothed on other clients. Each is a
// head followed by body segments laid along the head's trail, arching in and out of the water.
import { MeshBuilder, V3 } from './mesh';
import { sampleWater, newSample } from './waves';
import { clamp } from './world';

export const SEGS = 14, SEG_GAP = 2.6;
export const BOSS_POS = { x: 1900, y: -3300 };
export const segRadius = (i: number, scale = 1) => (i === 0 ? 2.3 : 2.0 * (1 - (i / SEGS) * 0.72)) * scale;
export enum MS { Approach, Circle, Lunge, Retreat, Dying }

export interface Monster {
  id: number; x: number; y: number; hd: number; sp: number; phase: number; hp: number; maxHp: number; st: MS; t: number;
  rear: number; dying: number; hurt: number; trail: number[]; segs: V3[]; orbit: number; dir: number;
  boss: boolean; scale: number; target: number; held: number;
  lx: number; ly: number; vx: number; vy: number; // measured velocity, sent to other clients
  // A rider with both hooks set steers the beast (world host only): turn -1..1, gas -1..1.
  rider: number; riderShip: number; steerT: number; steerTurn: number; steerGas: number; ramCd: number;
  dive: number; // 0..1 sounding into the deep (hooks flipped under its scales), then gone
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
  const segs: V3[] = [], gap = SEG_GAP * m.scale;
  let ax = m.x, ay = m.y, acc = 0, ti = 0;
  for (let i = 0; i < SEGS; i++) {
    const want = i * gap;
    while (acc < want && ti < m.trail.length) {
      const bx = m.trail[ti], by = m.trail[ti + 1], d = Math.hypot(bx - ax, by - ay);
      if (acc + d >= want) { const f = (want - acc) / Math.max(d, 1e-6); ax += (bx - ax) * f; ay += (by - ay) * f; acc = want; break; }
      acc += d; ax = bx; ay = by; ti += 2;
    }
    if (acc < want) { ax -= Math.cos(m.hd) * (want - acc); ay -= Math.sin(m.hd) * (want - acc); acc = want; }
    const w = sampleWater(ax, ay, tmp).z;
    const s = m.scale;
    const lift = m.rear * 7 * s * Math.max(0, 1 - i / 4);
    const und = 2.2 * Math.sqrt(s) * Math.sin(m.phase - i * 0.55) * (i === 0 ? 0.3 : 1);
    const sound = m.dive > 0 ? m.dive * m.dive * 55 * s * Math.min(1, 0.4 + (SEGS - i) / SEGS) : 0;
    segs.push([ax, ay, w - 0.5 * s + und + lift + (i === 0 ? 1.0 * s : 0) - m.dying * 9 * s - sound + 0.15 * Math.sin(time * 3 + i)]);
  }
  m.segs = segs;
}

export function pushTrail(m: Monster) {
  const lx = m.trail[0] ?? m.x - Math.cos(m.hd), ly = m.trail[1] ?? m.y - Math.sin(m.hd);
  const back = (m.x - lx) * Math.cos(m.hd) + (m.y - ly) * Math.sin(m.hd) < 0;
  if (back && m.trail.length > 2) {
    // Dragged backwards (by a harpoon line): slide back along the body instead of folding it.
    if (Math.hypot(m.trail[2] - m.x, m.trail[3] - m.y) < Math.hypot(lx - m.x, ly - m.y) + 0.2 * m.scale) m.trail.splice(0, 2);
    return;
  }
  if (Math.hypot(m.x - lx, m.y - ly) > 1.0 * m.scale) {
    m.trail.unshift(m.x, m.y);
    const max = Math.floor((SEGS * SEG_GAP * m.scale + 6) / m.scale) * 2 + 4;
    if (m.trail.length > max) m.trail.length = max;
  }
}

export interface ShipRef { id: number; x: number; y: number; psi: number; vx: number; vy: number; }
export interface MonsterEvents { ram: (m: Monster, shipId: number, side: number) => void; bossSpawn: (m: Monster) => void; }

function make(id: number, x: number, y: number, hd: number, boss: boolean): Monster {
  return {
    id, x, y, hd, sp: boss ? 12 : 9, phase: 0, hp: boss ? 40 : 4, maxHp: boss ? 40 : 4, st: MS.Approach, t: 0, rear: 0, dying: 0, hurt: 0,
    trail: [], segs: [], orbit: boss ? 110 + Math.random() * 30 : 45 + Math.random() * 30, dir: Math.random() < 0.5 ? 1 : -1,
    boss, scale: boss ? 3.2 : 1, target: 0, held: 0, rider: 0, riderShip: 0, steerT: 99, steerTurn: 0, steerGas: 0, ramCd: 0, dive: 0, lx: x, ly: y, vx: Math.cos(hd) * (boss ? 12 : 9), vy: Math.sin(hd) * (boss ? 12 : 9),
  };
}

export class MonsterSystem {
  list: Monster[] = [];
  bossCooldown = 0;
  private nextId = 1; private spawnT = 8;

  update(dt: number, ships: ShipRef[], stormAt: (x: number, y: number) => number, time: number, ev: MonsterEvents) {
    this.bossCooldown = Math.max(0, this.bossCooldown - dt);
    // Regular serpents: spawn near ships out in the storm.
    this.spawnT -= dt;
    if (this.spawnT <= 0 && ships.length) {
      this.spawnT = 8 + Math.random() * 10;
      const sh = ships[Math.floor(Math.random() * ships.length)];
      const storm = stormAt(sh.x, sh.y);
      const near = this.list.filter((m) => !m.boss && m.st !== MS.Dying && Math.hypot(m.x - sh.x, m.y - sh.y) < 500).length;
      if (storm >= 0.12 && near < 1 + Math.floor(storm * 2.6)) {
        const a = Math.random() * Math.PI * 2, d = 260 + Math.random() * 80;
        this.list.push(make(this.nextId++, sh.x + Math.cos(a) * d, sh.y + Math.sin(a) * d, a + Math.PI, false));
      }
    }
    // The Leviathan guards the heart of the storm.
    if (!this.list.some((m) => m.boss) && this.bossCooldown <= 0) {
      const sh = ships.find((s) => Math.hypot(s.x - BOSS_POS.x, s.y - BOSS_POS.y) < 900);
      if (sh) {
        const a = Math.random() * Math.PI * 2;
        const b = make(this.nextId++, sh.x + Math.cos(a) * 380, sh.y + Math.sin(a) * 380, a + Math.PI, true);
        this.list.push(b); ev.bossSpawn(b);
      }
    }
    for (const m of this.list) {
      // Actual velocity over the last frame, rope pulls included.
      if (dt > 0) { const kv = 1 - Math.exp(-dt * 6); m.vx += ((m.x - m.lx) / dt - m.vx) * kv; m.vy += ((m.y - m.ly) / dt - m.vy) * kv; }
      m.lx = m.x; m.ly = m.y;
      m.t += dt; m.hurt = Math.max(0, m.hurt - dt * 2);
      let tgt = ships[0];
      let best = Infinity;
      for (const s of ships) { const d = Math.hypot(m.x - s.x, m.y - s.y); if (d < best) { best = d; tgt = s; } }
      if (!tgt) { m.st = MS.Retreat; }
      const dx = tgt ? m.x - tgt.x : 0, dy = tgt ? m.y - tgt.y : 0, dist = tgt ? Math.hypot(dx, dy) : 1e9;
      m.target = tgt?.id ?? 0;
      const storm = tgt ? stormAt(tgt.x, tgt.y) : 0;
      let head = m.hd, speed = 8;
      const S = m.boss ? 1.5 : 1;
      if (m.st === MS.Dying) { m.dying = Math.min(1, m.dying + dt / (m.boss ? 5 : 3)); speed = 2; }
      else if (!m.boss && storm < 0.08 && m.st !== MS.Retreat) { m.st = MS.Retreat; m.t = 0; }
      if (m.st === MS.Approach && tgt) {
        head = Math.atan2(-dy, -dx); speed = 10 * S;
        if (dist < m.orbit + 15 * m.scale) { m.st = MS.Circle; m.t = 0; }
      } else if (m.st === MS.Circle && tgt) {
        const a = Math.atan2(dy, dx) + m.dir * 0.35;
        head = Math.atan2(tgt.y + Math.sin(a) * m.orbit - m.y, tgt.x + Math.cos(a) * m.orbit - m.x); speed = 8.5 * S;
        if (m.t > (m.boss ? 9 : 7) + (m.id % 5) * 1.5) { m.st = MS.Lunge; m.t = 0; }
      } else if (m.st === MS.Lunge && tgt) {
        const c = Math.cos(tgt.psi), s = Math.sin(tgt.psi);
        head = Math.atan2(tgt.y + tgt.vy * 2 - m.y, tgt.x + tgt.vx * 2 - m.x); speed = 15 * S;
        const lx = dx * c + dy * s, ly = -dx * s + dy * c;
        if (Math.abs(lx) < 19 + 2 * m.scale && Math.abs(ly) < 6 + 2 * m.scale) { ev.ram(m, tgt.id, Math.sign(ly) || 1); m.st = MS.Retreat; m.t = 0; }
        if (m.t > 12) { m.st = MS.Circle; m.t = 0; }
      } else if (m.st === MS.Retreat) {
        head = tgt ? Math.atan2(dy, dx) : m.hd; speed = 11 * S;
        if (m.t > (m.boss ? 4 : 5) && (storm >= 0.08 || m.boss)) { m.st = MS.Circle; m.t = 0; }
      }
      if (m.dive > 0) { m.dive = Math.min(1, m.dive + dt / 5); head = m.hd; speed = 16 * S; m.rear = 0; }
      // Ridden and steered: the rider's hooks override the hunt. Steer it into a ship to ram her.
      m.steerT += dt; m.ramCd -= dt;
      const steered = m.rider !== 0 && m.steerT < 0.6 && m.st !== MS.Dying && m.dive === 0;
      if (steered) {
        head = m.hd + m.steerTurn * 1.4; speed = (11 + 9 * m.steerGas) * S;
        if (m.st === MS.Lunge || m.st === MS.Approach) m.st = MS.Circle;
        m.t = 0;
        if (m.ramCd <= 0) for (const s of ships) {
          if (s.id === m.riderShip) continue;
          const c = Math.cos(s.psi), sn = Math.sin(s.psi), ex = m.x - s.x, ey = m.y - s.y;
          const lx = ex * c + ey * sn, ly = -ex * sn + ey * c;
          if (Math.abs(lx) < 19 + 2 * m.scale && Math.abs(ly) < 6 + 2 * m.scale) { ev.ram(m, s.id, Math.sign(ly) || 1); m.ramCd = 4; break; }
        }
      } else if (m.steerT > 2) m.rider = 0;
      // A harpoon rope slows the beast.
      if (m.held > 0) { speed *= 0.45; m.held = Math.max(0, m.held - dt); }
      const turn = Math.atan2(Math.sin(head - m.hd), Math.cos(head - m.hd));
      m.hd += clamp(turn, -(m.boss ? 0.6 : 0.9) * dt, (m.boss ? 0.6 : 0.9) * dt);
      m.sp += (speed - m.sp) * (1 - Math.exp(-dt));
      m.x += Math.cos(m.hd) * m.sp * dt; m.y += Math.sin(m.hd) * m.sp * dt;
      m.phase += dt * (1.6 + m.sp * 0.12) / Math.sqrt(m.scale);
      const wantRear = m.st === MS.Lunge && dist < 70 * m.scale ? 1 : m.st === MS.Circle ? 0.25 + 0.2 * Math.sin(m.t * 0.7) : 0;
      m.rear += (wantRear - m.rear) * (1 - Math.exp(-dt * 1.5));
      pushTrail(m);
      layoutSegments(m, time);
    }
    this.list = this.list.filter((m) => {
      if (m.dive >= 1) { if (m.boss) this.bossCooldown = 90; return false; }
      if (m.dying >= 1) { if (m.boss) this.bossCooldown = 180; return false; }
      const far = !ships.some((s) => Math.hypot(m.x - s.x, m.y - s.y) < (m.boss ? 1600 : 450));
      if (far && (m.st === MS.Retreat || m.boss) && !(m.rider && m.steerT < 2)) { if (m.boss) this.bossCooldown = 20; return false; }
      return true;
    });
  }

  // Hold a monster on a rope from a ship point: drag it back toward the ship when stretched.
  constrain(id: number, px: number, py: number, len: number, dt: number) {
    const m = this.list.find((x) => x.id === id);
    if (!m) return;
    m.held = 0.5;
    const dx = m.x - px, dy = m.y - py, d = Math.hypot(dx, dy);
    if (d > len && d > 1e-3) {
      const k = Math.min(1, dt * (m.boss ? 0.6 : 4)); // the Leviathan outweighs the ship: it tows her
      m.x -= dx / d * (d - len) * k; m.y -= dy / d * (d - len) * k;
      // The line hauls the head round toward the ship and saps its speed.
      const toShip = Math.atan2(-dy, -dx), turn = Math.atan2(Math.sin(toShip - m.hd), Math.cos(toShip - m.hd));
      if (Math.abs(turn) > Math.PI / 2) m.hd += Math.sign(turn) * Math.min(Math.abs(turn) - Math.PI / 2, dt * (m.boss ? 0.35 : 0.8) * (1 + (d - len) / 8));
      m.sp *= Math.exp(-dt * Math.min(d - len, 20) * 0.04);
    }
  }

  sound(id: number) { const m = this.list.find((x) => x.id === id); if (m && m.dive === 0) m.dive = 0.001; }

  steer(id: number, rider: number, riderShip: number, turn: number, gas: number) {
    const m = this.list.find((x) => x.id === id);
    if (!m) return;
    m.rider = rider; m.riderShip = riderShip; m.steerT = 0; m.steerTurn = clamp(turn, -1, 1); m.steerGas = clamp(gas, -1, 1);
  }

  hitTest(p: V3, pad: number): Monster | null {
    for (const m of this.list) {
      if (m.st === MS.Dying) continue;
      for (let i = 0; i < m.segs.length; i++) {
        const q = m.segs[i], r = segRadius(i, m.scale) + pad;
        if ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < r * r) return m;
      }
    }
    return null;
  }
  damage(m: Monster, dmg: number): boolean {
    m.hp -= dmg; m.hurt = 1;
    if (m.hp <= 0 && m.st !== MS.Dying) { m.st = MS.Dying; m.t = 0; return true; }
    if (!m.boss && (m.st === MS.Lunge || m.st === MS.Circle)) { m.st = MS.Retreat; m.t = 0; }
    return false;
  }
  pack(): number[][] {
    return this.list.map((m) => [m.id, +m.x.toFixed(2), +m.y.toFixed(2), +m.hd.toFixed(3), +m.phase.toFixed(3), m.hp, m.st, +m.rear.toFixed(3), +m.dying.toFixed(3), +m.hurt.toFixed(2), m.boss ? 1 : 0, m.maxHp, +m.sp.toFixed(2), +m.vx.toFixed(2), +m.vy.toFixed(2), +m.dive.toFixed(3)]);
  }
  // Other clients: dead-reckon from the world host's snapshot (taken at host time tm) and ease
  // toward it, advancing the body wave locally so motion stays smooth between 10 Hz updates.
  apply(rows: number[][], dt: number, time: number, tm: number) {
    const keep = new Map(this.list.map((m) => [m.id, m]));
    const k = 1 - Math.exp(-dt * 4), age = clamp(time - tm, 0, 0.5);
    this.list = rows.map((r) => {
      let m = keep.get(r[0]);
      const vx = r[13] ?? 0, vy = r[14] ?? 0, sp = r[12] ?? 0;
      if (!m) { m = make(r[0], r[1], r[2], r[3], !!r[10]); m.phase = r[4]; m.lx = r[1]; m.ly = r[2]; }
      m.vx = vx; m.vy = vy; m.sp = sp;
      const px = r[1] + vx * age, py = r[2] + vy * age;
      if (Math.hypot(px - m.x, py - m.y) > 60 * m.scale) { m.x = px; m.y = py; m.trail = []; }
      m.x += vx * dt + (px - m.x) * k; m.y += vy * dt + (py - m.y) * k;
      m.hd += Math.atan2(Math.sin(r[3] - m.hd), Math.cos(r[3] - m.hd)) * k;
      const rate = (1.6 + sp * 0.12) / Math.sqrt(m.scale);
      m.phase += rate * dt + (r[4] + rate * age - m.phase) * k;
      m.dive = r[15] ?? 0;
      m.hp = r[5]; m.st = r[6] as MS; m.rear += (r[7] - m.rear) * k; m.dying = r[8]; m.hurt = r[9]; m.maxHp = r[11] ?? m.maxHp;
      pushTrail(m); layoutSegments(m, time);
      return m;
    });
  }
}
