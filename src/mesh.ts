// Flat-shaded triangle soup: position (3) + colour (3). Normals come from screen derivatives,
// so winding does not matter. A colour with r > 1.5 is emissive (colour - 2).
export type V3 = [number, number, number];

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
