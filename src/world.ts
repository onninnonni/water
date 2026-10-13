// World layout and shared helpers. Units are metres; +x east, +y north, +z up.
export const IC = { x: 0, y: 0, R: 520 };       // island ring centre and radius
export const WIND_DIR = 35 * Math.PI / 180;      // direction the swell travels toward (NE)
export const SUN_DIR = norm3([-0.75, 0.35, 0.16]);

export const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;
export function smoothstep(e0: number, e1: number, x: number) {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
}
export function norm3(v: number[]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}
export function mix3(a: number[], b: number[], t: number): [number, number, number] {
  return [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)];
}

// 0 near the island, 1 out in the open storm.
export function stormAt(x: number, y: number) {
  return smoothstep(600, 3000, Math.hypot(x - IC.x, y - IC.y));
}
// 1 inside the sheltered bay, 0 outside the ring.
export function bayAt(x: number, y: number) {
  return 1 - smoothstep(IC.R - 140, IC.R - 30, Math.hypot(x - IC.x, y - IC.y));
}
