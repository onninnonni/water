// The F-15 Eagle: a low-poly model and an arcade flight model. Axes: x forward, y left, z up.
import { MeshBuilder, V3, cross, norm } from './mesh';

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

export function buildJetMesh(): number[] {
  const m = new MeshBuilder();
  const GREY: V3 = [0.46, 0.49, 0.53], DARK: V3 = [0.3, 0.32, 0.35], METAL: V3 = [0.2, 0.19, 0.19], GLASS: V3 = [0.06, 0.08, 0.11];
  // Two-sided flat panel (wings, stabilisers, fins).
  const panel = (a: V3, b: V3, c: V3, d: V3, col: V3, t = 0.08) => {
    const up = (p: V3): V3 => [p[0], p[1], p[2] + t], dn = (p: V3): V3 => [p[0], p[1], p[2] - t];
    m.quad(up(a), up(b), up(c), up(d), col); m.quad(dn(a), dn(b), dn(c), dn(d), col);
    m.quad(up(a), up(b), dn(b), dn(a), DARK); m.quad(up(c), up(d), dn(d), dn(c), DARK);
    m.quad(up(b), up(c), dn(c), dn(b), DARK);
  };
  // Fuselage, nose and canopy.
  m.tube([-8.2, 0, 0], [2.5, 0, 0.1], 1.05, 1.0, 12, GREY, true);
  m.tube([2.5, 0, 0.1], [6.5, 0, 0.15], 1.0, 0.7, 12, GREY);
  m.tube([6.5, 0, 0.15], [9.8, 0, 0.1], 0.7, 0.06, 12, DARK);
  m.sphere([4.6, 0, 0.95], 2.1, 0.62, 0.62, 12, GLASS);
  // Intakes and the twin engines.
  for (const s of [1, -1]) {
    m.box(-2.5, 3.6, s * 1.0, s * 2.15, -1.0, 0.55, GREY);
    m.box(3.5, 3.65, s * 1.15, s * 2.0, -0.9, 0.45, [0.04, 0.04, 0.05]);
    m.tube([-8.0, s * 0.78, -0.15], [-9.4, s * 0.78, -0.15], 0.66, 0.6, 10, METAL, true);
    m.tube([-9.38, s * 0.78, -0.15], [-9.45, s * 0.78, -0.15], 0.5, 0.5, 10, [2.4, 2.1, 1.9], true);
    // Swept wings, tailplanes and canted fins.
    panel([1.6, s * 1.9, 0], [-1.6, s * 6.6, -0.15], [-3.0, s * 6.6, -0.15], [-5.6, s * 1.9, 0], GREY);
    panel([-6.6, s * 1.4, -0.1], [-8.4, s * 4.4, -0.15], [-9.5, s * 4.4, -0.15], [-9.4, s * 1.4, -0.1], DARK, 0.06);
    const fin = (x: number, z: number): V3 => [x, s * (1.3 + z * 0.27), 0.6 + z];
    panel(fin(-4.8, 0), fin(-7.2, 3.3), fin(-8.6, 3.3), fin(-9.0, 0), DARK, 0.06);
    // Wingtip navigation lights: red to port (+y), green to starboard.
    m.box(-2.6, -2.2, s * 6.6, s * 6.75, -0.2, 0.0, s > 0 ? [2.9, 2.1, 2.1] : [2.1, 2.9, 2.2]);
  }
  return m.data;
}

const rot = (v: V3, k: V3, a: number): V3 => {
  // Rodrigues: rotate v about unit axis k by a.
  const c = Math.cos(a), s = Math.sin(a), d = (v[0] * k[0] + v[1] * k[1] + v[2] * k[2]) * (1 - c), x = cross(k, v);
  return [v[0] * c + x[0] * s + k[0] * d, v[1] * c + x[1] * s + k[1] * d, v[2] * c + x[2] * s + k[2] * d];
};

export class Jet {
  p: V3 = [0, 0, 500]; F: V3 = [1, 0, 0]; U: V3 = [0, 0, 1];
  sp = 180; thr = 0.7; ab = false; brake = false;

  place(x: number, y: number, z: number, yaw: number, sp = 190) {
    this.p = [x, y, z]; this.F = [Math.cos(yaw), Math.sin(yaw), 0]; this.U = [0, 0, 1]; this.sp = sp; this.thr = 0.7;
  }
  get R(): V3 { return norm(cross(this.F, this.U)); } // right wing
  get v(): V3 { const sink = Math.max(0, 95 - this.sp) * 0.35; return [this.F[0] * this.sp, this.F[1] * this.sp, this.F[2] * this.sp - sink]; }

  // pitch +1 = nose up, roll +1 = right wing down, yaw +1 = nose right.
  step(dt: number, pitch: number, roll: number, yaw: number) {
    let F = this.F, U = this.U, R = this.R;
    const auth = clamp(this.sp / 130, 0.3, 1);
    const pr = 1.35 * pitch * auth, rr = 2.2 * roll * auth, yr = 0.4 * yaw * auth;
    F = rot(F, R, pr * dt); U = rot(U, R, pr * dt);
    U = rot(U, F, rr * dt);
    R = norm(cross(F, U)); F = rot(F, U, -yr * dt);
    // Below flying speed the nose drops.
    if (this.sp < 90) { const a = (90 - this.sp) * 0.012 * dt; R = norm(cross(F, U)); F = rot(F, R, -a); U = rot(U, R, -a); }
    F = norm(F); R = norm(cross(F, U)); U = norm(cross(R, F));
    this.F = F; this.U = U;
    const thrust = 22 * this.thr + (this.ab ? 36 : 0) - (this.p[2] > 2500 ? (this.p[2] - 2500) * 0.05 : 0);
    const drag = 0.00035 * this.sp * this.sp * (this.brake ? 3.5 : 1);
    this.sp = Math.max(35, this.sp + (thrust - drag - 9.81 * F[2]) * dt);
    const v = this.v;
    this.p = [this.p[0] + v[0] * dt, this.p[1] + v[1] * dt, this.p[2] + v[2] * dt];
  }

  angles() {
    const F = this.F, R0 = norm(cross(F, [0, 0, 1])), U0 = cross(R0, F);
    return { yaw: Math.atan2(F[1], F[0]), pitch: Math.asin(clamp(F[2], -1, 1)), roll: Math.atan2(this.U[0] * R0[0] + this.U[1] * R0[1] + this.U[2] * R0[2], this.U[0] * U0[0] + this.U[1] * U0[1] + this.U[2] * U0[2]) };
  }
  // Forward and up axes from angles (for drawing other players' jets).
  static basis(yaw: number, pitch: number, roll: number): { F: V3; U: V3 } {
    const F: V3 = [Math.cos(pitch) * Math.cos(yaw), Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch)];
    const R0 = norm(cross(F, [0, 0, 1])), U0 = cross(R0, F);
    return { F, U: [U0[0] * Math.cos(roll) + R0[0] * Math.sin(roll), U0[1] * Math.cos(roll) + R0[1] * Math.sin(roll), U0[2] * Math.cos(roll) + R0[2] * Math.sin(roll)] };
  }
}
