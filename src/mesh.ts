// Flat-shaded triangle soup: position (3) + colour (3). Normals come from screen derivatives,
// so winding does not matter. A colour with r > 1.5 is emissive (colour - 2).
export type V3 = [number, number, number];

export const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const norm = (v: V3): V3 => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };

export class MeshBuilder {
  data: number[] = [];
  tri(a: V3, b: V3, c: V3, col: V3) { this.data.push(...a, ...col, ...b, ...col, ...c, ...col); }
  quad(a: V3, b: V3, c: V3, d: V3, col: V3) { this.tri(a, b, c, col); this.tri(a, c, d, col); }
  box(x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, col: V3, top: V3 = col) {
    this.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col);
    this.quad([x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1], col);
    this.quad([x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1], col);
    this.quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], col);
    this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], top);
    this.quad([x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], col);
  }
  // Frustum between two arbitrary points (masts, yards, rigging, barrels).
  tube(a: V3, b: V3, r0: number, r1: number, seg: number, col: V3, caps = false) {
    const d = norm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
    const ref: V3 = Math.abs(d[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    const u = norm(cross(d, ref)), v = cross(d, u);
    const ring = (c: V3, r: number, i: number): V3 => {
      const t = (i / seg) * Math.PI * 2, ct = Math.cos(t) * r, st = Math.sin(t) * r;
      return [c[0] + u[0] * ct + v[0] * st, c[1] + u[1] * ct + v[1] * st, c[2] + u[2] * ct + v[2] * st];
    };
    for (let i = 0; i < seg; i++) {
      this.quad(ring(a, r0, i), ring(a, r0, i + 1), ring(b, r1, i + 1), ring(b, r1, i), col);
      if (caps) { this.tri(ring(a, r0, i), ring(a, r0, i + 1), a, col); this.tri(ring(b, r1, i), ring(b, r1, i + 1), b, col); }
    }
  }
  sphere(c: V3, rx: number, ry: number, rz: number, seg: number, col: V3 | ((z: number) => V3)) {
    const cf = (z: number) => (typeof col === 'function' ? col(z) : col);
    const p = (i: number, j: number): V3 => {
      const th = (j / (seg / 2)) * Math.PI, ph = (i / seg) * Math.PI * 2;
      return [c[0] + Math.sin(th) * Math.cos(ph) * rx, c[1] + Math.sin(th) * Math.sin(ph) * ry, c[2] + Math.cos(th) * rz];
    };
    for (let j = 0; j < seg / 2; j++) for (let i = 0; i < seg; i++) {
      const a = p(i, j), b = p(i + 1, j), d = p(i + 1, j + 1), e = p(i, j + 1);
      this.quad(a, b, d, e, cf((a[2] + d[2]) / 2 - c[2]));
    }
  }
  // Vertical (possibly tapered) cylinder; colFn picks a colour per ring band.
  cyl(cx: number, cy: number, z0: number, z1: number, r0: number, r1: number, seg: number, col: V3 | ((zMid: number) => V3), cap = true) {
    const c = (z: number) => (typeof col === 'function' ? col(z) : col);
    for (let i = 0; i < seg; i++) {
      const a0 = (i / seg) * Math.PI * 2, a1 = ((i + 1) / seg) * Math.PI * 2;
      const p = (a: number, r: number, z: number): V3 => [cx + Math.cos(a) * r, cy + Math.sin(a) * r, z];
      this.quad(p(a0, r0, z0), p(a1, r0, z0), p(a1, r1, z1), p(a0, r1, z1), c((z0 + z1) / 2));
      if (cap) this.tri(p(a0, r1, z1), p(a1, r1, z1), [cx, cy, z1], c(z1));
    }
  }
}
