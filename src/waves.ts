// Deep-water Gerstner sea shared by the GPU (rendering) and the CPU (buoyancy).
// Long swell is huge out in the storm and nearly flat near the island; short chop stays a little.
import { stormAt, bayAt, smoothstep, WIND_DIR } from './world';

export const NW = 12;
const LAMBDA = [220, 160, 118, 92, 70, 52, 38, 27, 19, 13, 9, 6];
const STEEP = [0.068, 0.072, 0.078, 0.08, 0.08, 0.082, 0.075, 0.072, 0.07, 0.068, 0.064, 0.06];
const SPREAD = [0, 14, -19, 28, -33, 9, -12, 42, -47, 24, -27, 58];
const QTOT = 0.72;

export interface Wave { dx: number; dy: number; k: number; A: number; Q: number; w: number; phase: number; swell: number; L: number; }
let seed = 1234567;
const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
export const WAVES: Wave[] = LAMBDA.map((L, i) => {
  const k = 2 * Math.PI / L, A = STEEP[i] / k, a = WIND_DIR + SPREAD[i] * Math.PI / 180;
  // Deep-water dispersion: omega = sqrt(g k). Big waves are fast in m/s but slow for their size.
  return { dx: Math.cos(a), dy: Math.sin(a), k, A, Q: QTOT / (k * A * NW), w: Math.sqrt(9.81 * k), phase: rnd() * Math.PI * 2, swell: smoothstep(25, 70, L), L };
});

// Uniform arrays: A = (dir.x, dir.y, k, amplitude), B = (Q, phase at time t, swellness, wavelength).
export const uWA = new Float32Array(NW * 4);
export const uWB = new Float32Array(NW * 4);
WAVES.forEach((w, i) => { uWA.set([w.dx, w.dy, w.k, w.A], i * 4); uWB.set([w.Q, 0, w.swell, w.L], i * 4); });
const phaseT = new Float64Array(NW);
export function updatePhases(t: number) {
  for (let i = 0; i < NW; i++) {
    const w = WAVES[i];
    let p = (w.phase - w.w * t) % (2 * Math.PI);
    if (p < 0) p += 2 * Math.PI;
    phaseT[i] = p; uWB[i * 4 + 1] = p;
  }
}

// Amplitude multipliers for long swell and short chop at a rest position.
export function ampMuls(x: number, y: number): [number, number] {
  const s = stormAt(x, y), bay = bayAt(x, y);
  return [0.04 + 0.96 * s, (0.22 + 0.78 * s) * (1 - 0.55 * bay)];
}

export interface WSample { dx: number; dy: number; z: number; vx: number; vy: number; vz: number; }
export const newSample = (): WSample => ({ dx: 0, dy: 0, z: 0, vx: 0, vy: 0, vz: 0 });

function evalAt(x0: number, y0: number, o: WSample) {
  const [ml, ms] = ampMuls(x0, y0);
  o.dx = o.dy = o.z = o.vx = o.vy = o.vz = 0;
  for (let i = 0; i < NW; i++) {
    const w = WAVES[i];
    const A = w.A * (ms + (ml - ms) * w.swell);
    const th = w.k * (w.dx * x0 + w.dy * y0) + phaseT[i];
    const c = Math.cos(th), s = Math.sin(th), qa = w.Q * A;
    o.dx += qa * w.dx * c; o.dy += qa * w.dy * c; o.z += A * s;
    o.vx += qa * w.dx * w.w * s; o.vy += qa * w.dy * w.w * s; o.vz -= A * w.w * c;
  }
}

// Surface height and water velocity at a world position (inverts the horizontal displacement).
export function sampleWater(x: number, y: number, o: WSample) {
  let x0 = x, y0 = y;
  for (let i = 0; i < 3; i++) { evalAt(x0, y0, o); x0 = x - o.dx; y0 = y - o.dy; }
  evalAt(x0, y0, o);
  return o;
}

export function sigWaveHeight(x: number, y: number) {
  const [ml, ms] = ampMuls(x, y);
  let v = 0;
  for (const w of WAVES) { const a = w.A * (ms + (ml - ms) * w.swell); v += a * a / 2; }
  return 4 * Math.sqrt(v);
}
