import * as S from './shaders';
import { Mat4, perspective, lookAt, multiply } from './mat4';
import { IC, SUN_DIR, WIND_DIR, clamp, mix, mix3, smoothstep, stormAt, norm3 } from './world';
import { uWA, uWB, updatePhases, sampleWater, newSample, sigWaveHeight } from './waves';
import { buildLand, buildShoreMap, buildBuoy, PILLARS, ARCH, LIGHTHOUSE, BUOYS, SHORE, BUOY_COLS } from './island';
import { Boat, boatAxes, buildBoatMesh, buildWheelMesh, EYE, WHEEL_POS } from './boat';
import { SeaAudio } from './audio';
import type { V3 } from './mesh';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const isTouch = navigator.maxTouchPoints > 0 && matchMedia('(pointer: coarse)').matches;
const isMobile = isTouch || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
if (isTouch) document.body.classList.add('touch');

function fail(msg: string) {
  const e = $('err'); e.style.display = 'flex'; e.textContent = msg;
  $('start').style.display = 'none';
}

// ---------- WebGL ----------
const canvas = $<HTMLCanvasElement>('c');
const glc = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'high-performance' });
if (!glc) { fail('This game needs WebGL2, which this browser does not provide.'); throw new Error('no webgl2'); }
const G: WebGL2RenderingContext = glc;
G.getExtension('EXT_color_buffer_float');
G.getExtension('EXT_color_buffer_half_float');

function compile(type: number, src: string) {
  const s = G.createShader(type)!;
  G.shaderSource(s, src); G.compileShader(s);
  if (!G.getShaderParameter(s, G.COMPILE_STATUS)) throw new Error(G.getShaderInfoLog(s) ?? 'shader error');
  return s;
}
class Program {
  prog: WebGLProgram;
  u = new Map<string, WebGLUniformLocation>();
  constructor(vs: string, fs: string, attribs: string[]) {
    const p = G.createProgram()!;
    G.attachShader(p, compile(G.VERTEX_SHADER, vs)); G.attachShader(p, compile(G.FRAGMENT_SHADER, fs));
    attribs.forEach((a, i) => G.bindAttribLocation(p, i, a));
    G.linkProgram(p);
    if (!G.getProgramParameter(p, G.LINK_STATUS)) throw new Error(G.getProgramInfoLog(p) ?? 'link error');
    this.prog = p;
    const n = G.getProgramParameter(p, G.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < n; i++) {
      const name = G.getActiveUniform(p, i)!.name;
      this.u.set(name.replace(/\[0\]$/, ''), G.getUniformLocation(p, name)!);
    }
  }
  use() { G.useProgram(this.prog); return this; }
  f(n: string, v: number) { const l = this.u.get(n); if (l) G.uniform1f(l, v); return this; }
  i(n: string, v: number) { const l = this.u.get(n); if (l) G.uniform1i(l, v); return this; }
  v2(n: string, a: number, b: number) { const l = this.u.get(n); if (l) G.uniform2f(l, a, b); return this; }
  v3(n: string, v: ArrayLike<number>) { const l = this.u.get(n); if (l) G.uniform3f(l, v[0], v[1], v[2]); return this; }
  v4a(n: string, v: Float32Array) { const l = this.u.get(n); if (l) G.uniform4fv(l, v); return this; }
  v2a(n: string, v: Float32Array) { const l = this.u.get(n); if (l) G.uniform2fv(l, v); return this; }
  m4(n: string, m: Mat4) { const l = this.u.get(n); if (l) G.uniformMatrix4fv(l, false, m); return this; }
}

let pSky: Program, pOcean: Program, pSolid: Program, pSpray: Program, pRain: Program, pGlow: Program, pBolt: Program, pBeam: Program, pWake: Program, pShift: Program;
try {
  pSky = new Program(S.VS_QUAD, S.FS_SKY, ['aPos']);
  pOcean = new Program(S.VS_OCEAN, S.FS_OCEAN, ['aGrid']);
  pSolid = new Program(S.VS_SOLID, S.FS_SOLID, ['aPos', 'aCol']);
  pSpray = new Program(S.VS_SPRAY, S.FS_SPRAY, ['aP']);
  pRain = new Program(S.VS_RAIN, S.FS_RAIN, []);
  pGlow = new Program(S.VS_GLOW, S.FS_GLOW, ['aCenter', 'aCorner', 'aCol']);
  pBolt = new Program(S.VS_BOLT, S.FS_BOLT, ['aPos', 'aU']);
  pBeam = new Program(S.VS_BEAM, S.FS_BEAM, ['aP']);
  pWake = new Program(S.VS_QUAD, S.FS_WAKE, ['aPos']);
  pShift = new Program(S.VS_QUAD, S.FS_SHIFT, ['aPos']);
} catch (e) { fail('Could not start the graphics: ' + (e as Error).message); throw e; }

// ---------- Geometry ----------
function vao(layout: number[], data: Float32Array | null, usage: number = G.STATIC_DRAW) {
  const v = G.createVertexArray()!; G.bindVertexArray(v);
  const b = G.createBuffer()!; G.bindBuffer(G.ARRAY_BUFFER, b);
  if (data) G.bufferData(G.ARRAY_BUFFER, data, usage);
  const per = layout.reduce((a, n) => a + n, 0);
  let off = 0;
  layout.forEach((n, i) => { G.enableVertexAttribArray(i); G.vertexAttribPointer(i, n, G.FLOAT, false, per * 4, off); off += n * 4; });
  G.bindVertexArray(null);
  return { vao: v, buf: b, count: data ? data.length / per : 0 };
}
const quad = vao([2], new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
const solid = (d: number[]) => vao([3, 3], new Float32Array(d));

// Polar ocean grid centred on the camera, dense near and coarse toward the horizon.
const ocean = (() => {
  const NR = isMobile ? 120 : 170, NS = isMobile ? 144 : 200, RMAX = 7000, a = 6, ea = Math.exp(a) - 1;
  const verts = new Float32Array((NR + 1) * NS * 3);
  let k = 0;
  for (let i = 0; i <= NR; i++) {
    const t = i / NR, r = RMAX * (Math.exp(a * t) - 1) / ea, dr = RMAX * a * Math.exp(a * t) / ea / NR;
    const sp = Math.max(dr, 2 * Math.PI * r / NS);
    for (let j = 0; j < NS; j++) { const ang = (j / NS) * Math.PI * 2; verts[k++] = r * Math.cos(ang); verts[k++] = r * Math.sin(ang); verts[k++] = sp; }
  }
  const idx = new Uint32Array(NR * NS * 6);
  k = 0;
  for (let i = 0; i < NR; i++) for (let j = 0; j < NS; j++) {
    const p = i * NS + j, q = i * NS + (j + 1) % NS, r = (i + 1) * NS + j, s = (i + 1) * NS + (j + 1) % NS;
    idx[k++] = p; idx[k++] = q; idx[k++] = s; idx[k++] = p; idx[k++] = s; idx[k++] = r;
  }
  const o = vao([3], verts);
  G.bindVertexArray(o.vao);
  const ib = G.createBuffer()!; G.bindBuffer(G.ELEMENT_ARRAY_BUFFER, ib); G.bufferData(G.ELEMENT_ARRAY_BUFFER, idx, G.STATIC_DRAW);
  G.bindVertexArray(null);
  return { vao: o.vao, count: idx.length };
})();

const land = solid(buildLand());
const boatMesh = solid(buildBoatMesh());
const wheelMesh = solid(buildWheelMesh());
const buoyMeshes = [0, 1, 2].map((k) => solid(buildBuoy(k as 0 | 1 | 2)));

// Lighthouse beam: two long cones along +x and -x.
const beam = (() => {
  const L = 1800, Rb = L * Math.tan(3 * Math.PI / 180), seg = 16, d: number[] = [];
  for (const sgn of [1, -1]) for (let i = 0; i < seg; i++) {
    const a0 = i / seg * Math.PI * 2, a1 = (i + 1) / seg * Math.PI * 2;
    d.push(0, 0, 0, 0, sgn * L, Math.cos(a0) * Rb, Math.sin(a0) * Rb, 1, sgn * L, Math.cos(a1) * Rb, Math.sin(a1) * Rb, 1);
  }
  return vao([4], new Float32Array(d));
})();
const glowBuf = vao([3, 3, 4], null);
const boltBuf = vao([3, 1], null);
const MAXP = isMobile ? 500 : 900;
const sprayPos = new Float32Array(MAXP * 4), sprayVel = new Float32Array(MAXP * 3);
const sprayBuf = vao([4], sprayPos, G.DYNAMIC_DRAW);
const rainVao = G.createVertexArray()!;

// ---------- Textures ----------
const shoreData = buildShoreMap();
const shoreTex = G.createTexture()!;
G.bindTexture(G.TEXTURE_2D, shoreTex);
G.pixelStorei(G.UNPACK_ALIGNMENT, 1);
G.texImage2D(G.TEXTURE_2D, 0, G.R8, SHORE.res, SHORE.res, 0, G.RED, G.UNSIGNED_BYTE, shoreData);
G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, G.LINEAR); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, G.LINEAR);
G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.CLAMP_TO_EDGE); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.CLAMP_TO_EDGE);

interface RT { tex: WebGLTexture; fb: WebGLFramebuffer; }
function makeRT(size: number, internal: number, type: number): RT | null {
  const tex = G.createTexture()!;
  G.bindTexture(G.TEXTURE_2D, tex);
  G.texImage2D(G.TEXTURE_2D, 0, internal, size, size, 0, G.RGBA, type, null);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, G.LINEAR); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, G.LINEAR);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.CLAMP_TO_EDGE); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.CLAMP_TO_EDGE);
  const fb = G.createFramebuffer()!;
  G.bindFramebuffer(G.FRAMEBUFFER, fb);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT0, G.TEXTURE_2D, tex, 0);
  const ok = G.checkFramebufferStatus(G.FRAMEBUFFER) === G.FRAMEBUFFER_COMPLETE;
  G.clearColor(0, 0, 0, 0); G.clear(G.COLOR_BUFFER_BIT);
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  return ok ? { tex, fb } : null;
}
const WAKE = { size: 256, res: 256, ox: 0, oy: -360 };
const mkWake = () => makeRT(WAKE.res, G.RGBA16F, G.HALF_FLOAT) ?? makeRT(WAKE.res, G.RGBA8, G.UNSIGNED_BYTE)!;
let wakeA = mkWake(), wakeB = mkWake();

// ---------- Small matrix helpers ----------
function fromAxes(X: V3, Y: V3, Z: V3, p: V3): Mat4 {
  return new Float32Array([X[0], X[1], X[2], 0, Y[0], Y[1], Y[2], 0, Z[0], Z[1], Z[2], 0, p[0], p[1], p[2], 1]);
}
function rigidInverse(X: V3, Y: V3, Z: V3, p: V3): Mat4 {
  const d = (a: V3) => -(a[0] * p[0] + a[1] * p[1] + a[2] * p[2]);
  return new Float32Array([X[0], Y[0], Z[0], 0, X[1], Y[1], Z[1], 0, X[2], Y[2], Z[2], 0, d(X), d(Y), d(Z), 1]);
}
const IDENT = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

// ---------- Game state ----------
const boat = new Boat();
const audio = new SeaAudio();
let playing = false;
let time = 0;
let camMode: 'bridge' | 'chase' = 'bridge';
const atm = { s: stormAt(boat.x, boat.y), fog: [0, 0, 0] as V3, sky: [0, 0, 0] as V3, sun: [0, 0, 0] as V3, ambT: [0, 0, 0] as V3, ambB: [0, 0, 0] as V3, fogDen: 0, rain: 0, wind: 0 };
const windVec: [number, number] = [Math.cos(WIND_DIR), Math.sin(WIND_DIR)];

function updateAtmosphere(dt: number) {
  atm.s += (stormAt(boat.x, boat.y) - atm.s) * (1 - Math.exp(-dt / 4));
  const a = 0.15 + 0.85 * atm.s;
  atm.fog = mix3([0.3, 0.34, 0.37], [0.13, 0.155, 0.18], a);
  atm.sky = mix3([0.19, 0.24, 0.3], [0.045, 0.055, 0.07], a);
  atm.sun = mix3([0.62, 0.5, 0.34], [0.08, 0.08, 0.09], a);
  atm.ambT = mix3([0.3, 0.34, 0.38], [0.13, 0.15, 0.18], a);
  atm.ambB = [atm.ambT[0] * 0.4, atm.ambT[1] * 0.4, atm.ambT[2] * 0.45];
  atm.fogDen = mix(0.00045, 0.0017, a);
  atm.rain = smoothstep(0.05, 1, a);
  atm.wind = mix(0.2, 1, a);
}

// ---------- Input ----------
const keys = new Set<string>();
let touchRudder: number | null = null;
const look = { yaw: 0, pitch: 0, active: false };
interface Ptr { zone: 'steer' | 'throttle' | 'look'; x0: number; y0: number; x: number; y: number; lever0: number; }
const ptrs = new Map<number, Ptr>();
window.addEventListener('keydown', (e) => {
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keys.add(k);
  if (!playing) return;
  if (k === 'c') toggleCam();
  if (k === 'm') toggleMute();
  if (k === 'x' || k === ' ') boat.lever = 0;
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', ' '].includes(k)) e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));
window.addEventListener('blur', () => keys.clear());
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  if (!playing) return;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  const zone = e.pointerType === 'mouse' ? 'look' : e.clientX < innerWidth / 2 ? 'steer' : 'throttle';
  ptrs.set(e.pointerId, { zone, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, lever0: boat.lever });
  if (zone === 'look') look.active = true;
  document.body.classList.remove('hint');
});
canvas.addEventListener('pointermove', (e) => {
  const p = ptrs.get(e.pointerId);
  if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (p.zone === 'look') { look.yaw = clamp(look.yaw - dx * 0.005, -2.6, 2.6); look.pitch = clamp(look.pitch - dy * 0.004, -0.7, 0.6); }
  else if (p.zone === 'steer') touchRudder = clamp(-(p.x - p.x0) / (innerWidth * 0.14), -1, 1);
  else boat.lever = clamp(p.lever0 - (p.y - p.y0) / (innerHeight * 0.32), -0.5, 1);
});
const endPtr = (e: PointerEvent) => {
  const p = ptrs.get(e.pointerId);
  if (!p) return;
  if (p.zone === 'steer') touchRudder = null;
  if (p.zone === 'look') look.active = false;
  ptrs.delete(e.pointerId);
};
canvas.addEventListener('pointerup', endPtr);
canvas.addEventListener('pointercancel', endPtr);

function updateControls(dt: number) {
  const up = keys.has('w') || keys.has('ArrowUp'), down = keys.has('s') || keys.has('ArrowDown');
  if (up) boat.lever = Math.min(1, boat.lever + 0.5 * dt);
  if (down) boat.lever = Math.max(-0.5, boat.lever - 0.5 * dt);
  const kb = (keys.has('a') || keys.has('ArrowLeft') ? 1 : 0) - (keys.has('d') || keys.has('ArrowRight') ? 1 : 0);
  const target = touchRudder ?? kb;
  const rate = target === 0 ? 3 : touchRudder !== null ? 8 : 2.2;
  boat.rudder += clamp(target - boat.rudder, -rate * dt, rate * dt);
  if (!look.active) { look.yaw *= Math.exp(-dt * 2.2); look.pitch *= Math.exp(-dt * 2.2); }
}

// ---------- Camera ----------
const cam = { eye: [0, 0, 5] as V3, fwd: [1, 0, 0] as V3, right: [0, -1, 0] as V3, up: [0, 0, 1] as V3, fov: 1.2, near: 0.6, shake: 0 };
const chase = { pos: [0, -330, 12] as V3, yaw: -Math.PI / 2 };
const camSample = newSample();
function updateCamera(dt: number) {
  cam.shake *= Math.exp(-dt * 5);
  const jit = () => (Math.random() - 0.5) * cam.shake;
  const sp01 = Math.min(Math.hypot(boat.vx, boat.vy) / 11, 1);
  if (!playing) {
    // Title: a slow orbit around the boat in the bay, looking out toward the stacks.
    const a = Math.PI / 2 + time * 0.04, Rr = 30;
    cam.eye = [boat.x + Math.cos(a) * Rr, boat.y + Math.sin(a) * Rr, boat.z + 7];
    const tgt: V3 = [boat.x, boat.y - 25, boat.z + 4];
    cam.fwd = norm3(sub(tgt, cam.eye)); cam.right = norm3(cross(cam.fwd, [0, 0, 1])); cam.up = cross(cam.right, cam.fwd);
    cam.fov = 55 * Math.PI / 180; cam.near = 1;
    return;
  }
  if (camMode === 'bridge') {
    const ax = boatAxes(boat.psi, boat.th, boat.ph);
    cam.eye = [
      boat.x + ax.X[0] * EYE[0] + ax.Y[0] * EYE[1] + ax.Z[0] * EYE[2] + jit(),
      boat.y + ax.X[1] * EYE[0] + ax.Y[1] * EYE[1] + ax.Z[1] * EYE[2] + jit(),
      boat.z + ax.X[2] * EYE[0] + ax.Y[2] * EYE[1] + ax.Z[2] * EYE[2] + jit(),
    ];
    // The head stabilises a little against the hull's motion and glances into turns.
    const c = boatAxes(boat.psi + look.yaw + boat.r * 0.35, boat.th * 0.8 - 0.07 + look.pitch + jit() * 0.3, boat.ph * 0.78);
    cam.fwd = c.X; cam.up = c.Z; cam.right = [-c.Y[0], -c.Y[1], -c.Y[2]];
    cam.fov = (68 + 6 * sp01) * Math.PI / 180; cam.near = 0.6;
  } else {
    chase.yaw += Math.atan2(Math.sin(boat.psi - chase.yaw), Math.cos(boat.psi - chase.yaw)) * (1 - Math.exp(-dt * 1.6));
    const yaw = chase.yaw + look.yaw, dist = 30, h = 9 - look.pitch * 12;
    const want: V3 = [boat.x - Math.cos(yaw) * dist, boat.y - Math.sin(yaw) * dist, boat.z + h];
    const kxy = 1 - Math.exp(-dt * 3), kz = 1 - Math.exp(-dt * 2);
    chase.pos = [chase.pos[0] + (want[0] - chase.pos[0]) * kxy, chase.pos[1] + (want[1] - chase.pos[1]) * kxy, chase.pos[2] + (want[2] - chase.pos[2]) * kz];
    const wz = sampleWater(chase.pos[0], chase.pos[1], camSample).z;
    chase.pos[2] = Math.max(chase.pos[2], wz + 3);
    cam.eye = [chase.pos[0] + jit(), chase.pos[1] + jit(), chase.pos[2] + jit()];
    const tgt: V3 = [boat.x + Math.cos(boat.psi) * 8, boat.y + Math.sin(boat.psi) * 8, boat.z + 2.5];
    cam.fwd = norm3(sub(tgt, cam.eye)); cam.right = norm3(cross(cam.fwd, [0, 0, 1])); cam.up = cross(cam.right, cam.fwd);
    cam.fov = (60 + 6 * sp01) * Math.PI / 180; cam.near = 1;
  }
}
function toggleCam() {
  camMode = camMode === 'bridge' ? 'chase' : 'bridge';
  $('camBtn').textContent = camMode === 'bridge' ? 'Chase cam' : 'Bridge cam';
  chase.pos = [boat.x - Math.cos(boat.psi) * 30, boat.y - Math.sin(boat.psi) * 30, boat.z + 9];
  chase.yaw = boat.psi;
}
function toggleMute() { audio.setMuted(!audio.muted); $('muteBtn').textContent = audio.muted ? 'Sound' : 'Mute'; }

// ---------- Spray ----------
let sprayNext = 0;
function emit(p: V3, v: V3, life = 1) {
  const i = sprayNext; sprayNext = (sprayNext + 1) % MAXP;
  sprayPos[i * 4] = p[0]; sprayPos[i * 4 + 1] = p[1]; sprayPos[i * 4 + 2] = p[2]; sprayPos[i * 4 + 3] = life;
  sprayVel[i * 3] = v[0]; sprayVel[i * 3 + 1] = v[1]; sprayVel[i * 3 + 2] = v[2];
}
function emitLocal(l: V3, lv: V3, life = 1) {
  const ax = boatAxes(boat.psi, boat.th, boat.ph);
  const w = (a: V3): V3 => [ax.X[0] * a[0] + ax.Y[0] * a[1] + ax.Z[0] * a[2], ax.X[1] * a[0] + ax.Y[1] * a[1] + ax.Z[1] * a[2], ax.X[2] * a[0] + ax.Y[2] * a[1] + ax.Z[2] * a[2]];
  const p = w(l), v = w(lv);
  emit([boat.x + p[0], boat.y + p[1], boat.z + p[2]], [boat.vx + v[0], boat.vy + v[1], boat.vz + v[2]], life);
}
const R = () => Math.random();
function updateSpray(dt: number, inv: Mat4) {
  const drag = Math.exp(-dt * 0.7);
  for (let i = 0; i < MAXP; i++) {
    if (sprayPos[i * 4 + 3] <= 0) continue;
    sprayVel[i * 3 + 2] -= 9.81 * dt;
    sprayVel[i * 3] *= drag; sprayVel[i * 3 + 1] *= drag;
    const x = sprayPos[i * 4] += sprayVel[i * 3] * dt;
    const y = sprayPos[i * 4 + 1] += sprayVel[i * 3 + 1] * dt;
    const z = sprayPos[i * 4 + 2] += sprayVel[i * 3 + 2] * dt;
    let life = sprayPos[i * 4 + 3] - dt / 1.7;
    if (z < boat.z - 5) life = 0;
    const lx = inv[0] * x + inv[4] * y + inv[8] * z + inv[12], ly = inv[1] * x + inv[5] * y + inv[9] * z + inv[13], lz = inv[2] * x + inv[6] * y + inv[10] * z + inv[14];
    if (lx > -3.7 && lx < 0.45 && Math.abs(ly) < 1.85 && lz > 0.9 && lz < 4.1) life = 0;
    sprayPos[i * 4 + 3] = life;
  }
}

// ---------- Lightning ----------
const bolt = { t: 99, I: 0, pts: [] as V3[][], mid: [0, 0, 500] as V3 };
let nextStrike = 5, flash = 0;
const flashDir: V3 = [0, 0, 1];
function jagged(a: V3, b: V3, levels: number, rough: number): V3[] {
  let pts: V3[] = [a, b];
  for (let l = 0; l < levels; l++) {
    const np: V3[] = [pts[0]];
    for (let k = 0; k < pts.length - 1; k++) {
      const p = pts[k], q = pts[k + 1], len = Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
      np.push([(p[0] + q[0]) / 2 + (R() - 0.5) * len * rough, (p[1] + q[1]) / 2 + (R() - 0.5) * len * rough, (p[2] + q[2]) / 2 + (R() - 0.5) * len * 0.15], q);
    }
    pts = np;
  }
  return pts;
}
function strike() {
  const away = Math.atan2(boat.y - IC.y, boat.x - IC.x) + (R() - 0.5) * 2.6;
  const dist = mix(2600, 900, atm.s) + R() * mix(4200, 3000, atm.s);
  const x = cam.eye[0] + Math.cos(away) * dist, y = cam.eye[1] + Math.sin(away) * dist;
  const main = jagged([x, y, 1150], [x + (R() - 0.5) * 300, y + (R() - 0.5) * 300, 0], 7, 0.55);
  bolt.pts = [main];
  for (let b = 0; b < 3; b++) {
    const s = main[Math.floor(R() * main.length * 0.6)];
    bolt.pts.push(jagged(s, [s[0] + (R() - 0.5) * 500, s[1] + (R() - 0.5) * 500, s[2] - 200 - R() * 300], 5, 0.5));
  }
  bolt.mid = [x, y, 550]; bolt.t = 0;
  bolt.I = clamp(2.4 / (1 + dist / 1500), 0.35, 1.7);
  audio.thunder(dist);
}
function updateLightning(dt: number) {
  nextStrike -= dt;
  if (nextStrike <= 0) { strike(); nextStrike = mix(24, 3.5, atm.s) * (0.45 + R()); }
  bolt.t += dt;
  const t = bolt.t;
  const f = t < 0.05 ? 1 : t < 0.11 ? 0.25 : t < 0.19 ? 0.85 : t < 0.25 ? 0.2 : 0.55 * Math.exp(-(t - 0.25) * 7);
  flash = bolt.I * f;
  const d = norm3(sub(bolt.mid, cam.eye)); flashDir[0] = d[0]; flashDir[1] = d[1]; flashDir[2] = d[2];
}

// ---------- Wake map ----------
const prevMarks = { stern: [0, 0] as number[], port: [0, 0] as number[], stbd: [0, 0] as number[] };
let wakeBlob = 0;
const segU = new Float32Array(12), segP = new Float32Array(6);
function wakePass(p: Program) {
  G.bindFramebuffer(G.FRAMEBUFFER, wakeB.fb); G.viewport(0, 0, WAKE.res, WAKE.res);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  p.i('uSrc', 0);
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  [wakeA, wakeB] = [wakeB, wakeA];
}
function updateWake() {
  const dx = Math.round(boat.x - WAKE.ox), dy = Math.round(boat.y - WAKE.oy);
  if (Math.abs(dx) > 48 || Math.abs(dy) > 48) {
    wakePass(pShift.use().v2('uShift', dx / WAKE.size, dy / WAKE.size));
    WAKE.ox += dx; WAKE.oy += dy;
  }
  const uv = (p: number[]) => [(p[0] - WAKE.ox) / WAKE.size + 0.5, (p[1] - WAKE.oy) / WAKE.size + 0.5];
  const st = boat.local(-7.2, 0, 0), po = boat.local(4.6, 2.3, 0), sb = boat.local(4.6, -2.3, 0);
  const speed = Math.hypot(boat.vx, boat.vy);
  const marks: [number[], V3, number, number][] = [
    [prevMarks.stern, st, 1.8 + speed * 0.08, 0.012 * clamp(Math.abs(boat.rpm) * 0.8 + speed / 11, 0, 1.3)],
    [prevMarks.port, po, 0.9, 0.012 * smoothstep(2.5, 10, speed)],
    [prevMarks.stbd, sb, 0.9, 0.012 * smoothstep(2.5, 10, speed)],
  ];
  marks.forEach(([prev, now, w, I], i) => {
    const a = uv(prev), b = uv(now);
    segU.set([a[0], a[1], b[0], b[1]], i * 4); segP.set([w / WAKE.size, I], i * 2);
    prev[0] = now[0]; prev[1] = now[1];
  });
  const bow = uv(boat.local(6, 0, 0));
  wakePass(pWake.use().v2('uTexel', 1 / WAKE.res, 1 / WAKE.res).v4a('uSeg', segU).v2a('uSegP', segP)
    .v3('uBlob', [bow[0], bow[1], 5 / WAKE.size]).f('uBlobI', wakeBlob));
  wakeBlob = 0;
}
function resetWakeMarks() {
  const st = boat.local(-7.2, 0, 0), po = boat.local(4.6, 2.3, 0), sb = boat.local(4.6, -2.3, 0);
  prevMarks.stern = [st[0], st[1]]; prevMarks.port = [po[0], po[1]]; prevMarks.stbd = [sb[0], sb[1]];
}
resetWakeMarks();

// ---------- Buoys ----------
const buoySample = newSample(), buoyGx = newSample(), buoyGy = newSample();
const buoyModels: Mat4[] = BUOYS.map(() => IDENT);
const buoyTops: V3[] = BUOYS.map(() => [0, 0, 0]);
function updateBuoys() {
  BUOYS.forEach((b, i) => {
    const z = sampleWater(b.x, b.y, buoySample).z;
    const gx = (sampleWater(b.x + 1.5, b.y, buoyGx).z - z) / 1.5, gy = (sampleWater(b.x, b.y + 1.5, buoyGy).z - z) / 1.5;
    const ax = boatAxes(b.phase, Math.atan(gx * 0.8), -Math.atan(gy * 0.8));
    const p: V3 = [b.x, b.y, z - 0.2];
    buoyModels[i] = fromAxes(ax.X, ax.Y, ax.Z, p);
    buoyTops[i] = [p[0] + ax.Z[0] * 3.7, p[1] + ax.Z[1] * 3.7, p[2] + ax.Z[2] * 3.7];
  });
}

// ---------- Glows ----------
const MAXG = 64;
const glowData = new Float32Array(MAXG * 60);
let glowCount = 0;
const CORNERS = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]];
function addGlow(p: V3, size: number, col: V3, a: number) {
  if (a <= 0.002 || glowCount >= MAXG) return;
  CORNERS.forEach((c, k) => glowData.set([p[0], p[1], p[2], c[0], c[1], size, col[0], col[1], col[2], a], glowCount * 60 + k * 10));
  glowCount++;
}
const BUOY_LIGHT: V3[] = [[1, 0.25, 0.2], [0.3, 1, 0.45], [1, 0.8, 0.3]];

// ---------- HUD ----------
const compass = $<HTMLCanvasElement>('compass'), cctx = compass.getContext('2d')!;
const minimap = $<HTMLCanvasElement>('minimap'), mctx = minimap.getContext('2d')!;
const islandImg = (() => {
  const res = SHORE.res, out = document.createElement('canvas');
  out.width = res; out.height = res;
  const ctx = out.getContext('2d')!, img = ctx.createImageData(res, res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const h = shoreData[j * res + i] / 255 * 40 - 20, o = ((res - 1 - j) * res + i) * 4;
    if (h > 0.3) img.data.set([118, 126, 112, 255], o);
    else if (h > -8) img.data.set([60, 98, 104, Math.round(200 * (1 + h / 8))], o);
  }
  ctx.putImageData(img, 0, 0);
  return out;
})();
const trail: number[][] = [];
let trailT = 0;
const knotsEl = $('knots'), seaEl = $('sea'), leverFill = $('leverFill'), leverZero = $('leverZero'), needle = $('rudderNeedle'), toastEl = $('toast');
let toastTimer = 0;
function toast(msg: string, dur = 3.5) { toastEl.textContent = msg; toastEl.classList.add('show'); toastTimer = dur; }
const bearing = (dx: number, dy: number) => ((Math.atan2(dx, dy) * 180 / Math.PI) + 360) % 360;
const SEA_STATES = ['Sheltered water', 'Moderate swell', 'Rough sea', 'Very rough', 'High seas'];
let seaState = 0, seaCand = 0, seaCandT = 0;

function drawCompass(hdg: number) {
  const W = compass.width, Hh = compass.height;
  if (!W) return;
  const dpr = W / compass.clientWidth, ppd = W / 120;
  cctx.clearRect(0, 0, W, Hh);
  const g = cctx.createLinearGradient(0, 0, W, 0);
  g.addColorStop(0, 'rgba(8,14,20,0)'); g.addColorStop(0.15, 'rgba(8,14,20,0.45)'); g.addColorStop(0.85, 'rgba(8,14,20,0.45)'); g.addColorStop(1, 'rgba(8,14,20,0)');
  cctx.fillStyle = g; cctx.fillRect(0, 0, W, Hh);
  cctx.textAlign = 'center'; cctx.textBaseline = 'top';
  const names: Record<number, string> = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
  for (let d = Math.ceil((hdg - 60) / 5) * 5; d <= hdg + 60; d += 5) {
    const x = W / 2 + (d - hdg) * ppd, dd = ((d % 360) + 360) % 360;
    cctx.globalAlpha = Math.max(1 - Math.abs(d - hdg) / 62, 0);
    cctx.fillStyle = '#e4ebf0';
    const major = dd % 15 === 0;
    cctx.fillRect(x - dpr * 0.5, Hh - (major ? 9 : 5) * dpr, dpr, (major ? 9 : 5) * dpr);
    if (names[dd] !== undefined) { cctx.font = `600 ${12 * dpr}px system-ui`; cctx.fillText(names[dd], x, 3 * dpr); }
    else if (major) { cctx.font = `${9 * dpr}px system-ui`; cctx.fillStyle = 'rgba(228,235,240,0.65)'; cctx.fillText(String(dd), x, 5 * dpr); }
  }
  cctx.globalAlpha = 1;
  // Lighthouse bearing marker.
  const lb = bearing(LIGHTHOUSE.x - boat.x, LIGHTHOUSE.y - boat.y);
  let rel = ((lb - hdg + 540) % 360) - 180;
  const edge = Math.abs(rel) > 58;
  rel = clamp(rel, -58, 58);
  const mx = W / 2 + rel * ppd;
  cctx.fillStyle = '#e9b44c';
  cctx.beginPath();
  if (edge) { const s = Math.sign(rel); cctx.moveTo(mx + s * 6 * dpr, Hh / 2); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 - 6 * dpr); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 + 6 * dpr); }
  else { cctx.moveTo(mx, Hh - 11 * dpr); cctx.lineTo(mx - 5 * dpr, Hh - 19 * dpr); cctx.lineTo(mx + 5 * dpr, Hh - 19 * dpr); }
  cctx.fill();
  cctx.fillStyle = '#e4ebf0';
  cctx.fillRect(W / 2 - dpr, Hh - 14 * dpr, 2 * dpr, 14 * dpr);
}

function drawMinimap() {
  const W = minimap.width;
  if (!W) return;
  const dpr = W / minimap.clientWidth, cx = W / 2, cy = W / 2, rad = W / 2 - 2 * dpr;
  const scale = rad / 1300;
  const mx = (x: number) => cx + (x - boat.x) * scale, my = (y: number) => cy - (y - boat.y) * scale;
  mctx.clearRect(0, 0, W, W);
  mctx.save();
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.clip();
  const g = mctx.createRadialGradient(mx(IC.x), my(IC.y), 600 * scale, mx(IC.x), my(IC.y), 3000 * scale);
  g.addColorStop(0, 'rgba(34,62,72,0.88)'); g.addColorStop(1, 'rgba(8,13,18,0.92)');
  mctx.fillStyle = g; mctx.fillRect(0, 0, W, W);
  mctx.setLineDash([4 * dpr, 4 * dpr]); mctx.strokeStyle = 'rgba(233,180,76,0.45)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(mx(IC.x), my(IC.y), 1500 * scale, 0, Math.PI * 2); mctx.stroke(); mctx.setLineDash([]);
  const sz = SHORE.size * scale;
  mctx.drawImage(islandImg, mx(IC.x + SHORE.min), my(IC.y + SHORE.min + SHORE.size), sz, sz);
  mctx.fillStyle = '#9aa196';
  for (const p of PILLARS) { mctx.beginPath(); mctx.arc(mx(p.x), my(p.y), Math.max(p.r * scale, 1.5 * dpr), 0, Math.PI * 2); mctx.fill(); }
  BUOYS.forEach((b) => { const c = BUOY_COLS[b.kind]; mctx.fillStyle = `rgb(${c.map((v) => Math.min(255, Math.round(v * 320))).join(',')})`; mctx.fillRect(mx(b.x) - dpr, my(b.y) - dpr, 2 * dpr, 2 * dpr); });
  const blink = Math.sin(time * 1.1) > 0.6 ? 1 : 0.55;
  mctx.fillStyle = `rgba(255,214,120,${blink})`; mctx.beginPath(); mctx.arc(mx(LIGHTHOUSE.x), my(LIGHTHOUSE.y), 3 * dpr, 0, Math.PI * 2); mctx.fill();
  mctx.lineWidth = 1.5 * dpr;
  for (let i = 1; i < trail.length; i++) {
    mctx.strokeStyle = `rgba(200,225,235,${(i / trail.length) * 0.6})`;
    mctx.beginPath(); mctx.moveTo(mx(trail[i - 1][0]), my(trail[i - 1][1])); mctx.lineTo(mx(trail[i][0]), my(trail[i][1])); mctx.stroke();
  }
  mctx.restore();
  mctx.strokeStyle = 'rgba(228,235,240,0.35)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.stroke();
  mctx.fillStyle = '#e4ebf0'; mctx.font = `600 ${10 * dpr}px system-ui`; mctx.textAlign = 'center'; mctx.fillText('N', cx, 13 * dpr);
  mctx.save(); mctx.translate(cx, cy); mctx.rotate(-boat.psi);
  mctx.fillStyle = '#e9b44c'; mctx.beginPath(); mctx.moveTo(8 * dpr, 0); mctx.lineTo(-5 * dpr, 4.5 * dpr); mctx.lineTo(-3 * dpr, 0); mctx.lineTo(-5 * dpr, -4.5 * dpr); mctx.fill();
  mctx.restore();
}

let gatePrev = [boat.x, boat.y];
function checkGates() {
  for (let i = 0; i < PILLARS.length - 1; i++) {
    const A = PILLARS[i], B = PILLARS[i + 1];
    const ex = B.x - A.x, ey = B.y - A.y;
    const s0 = ex * (gatePrev[1] - A.y) - ey * (gatePrev[0] - A.x), s1 = ex * (boat.y - A.y) - ey * (boat.x - A.x);
    if (s0 * s1 >= 0) continue;
    const t = ((boat.x - A.x) * ex + (boat.y - A.y) * ey) / (ex * ex + ey * ey);
    if (t < 0 || t > 1) continue;
    const islandSide = ex * (IC.y - A.y) - ey * (IC.x - A.x);
    const outward = (s1 > 0) !== (islandSide > 0);
    toast(i === ARCH.a ? (outward ? 'Through the arch' : 'Home through the arch') : outward ? 'Past the stacks' : 'Back inside the bay');
  }
  gatePrev = [boat.x, boat.y];
}

function updateHud(dt: number) {
  const sp = Math.hypot(boat.vx, boat.vy);
  knotsEl.textContent = (sp * 1.944).toFixed(0);
  const hs = sigWaveHeight(boat.x, boat.y);
  const s = stormAt(boat.x, boat.y);
  const cat = s < 0.06 ? 0 : s < 0.28 ? 1 : s < 0.52 ? 2 : s < 0.78 ? 3 : 4;
  if (cat !== seaCand) { seaCand = cat; seaCandT = 0; } else seaCandT += dt;
  if (seaCand !== seaState && seaCandT > 1.5) { seaState = seaCand; if (playing) toast(SEA_STATES[seaState]); }
  seaEl.innerHTML = `${SEA_STATES[seaState]} · waves <b>${hs.toFixed(1)} m</b>`;
  const zero = 0.5 / 1.5 * 100;
  leverZero.style.bottom = `${zero}%`;
  if (boat.lever >= 0) { leverFill.style.bottom = `${zero}%`; leverFill.style.height = `${boat.lever / 1.5 * 100}%`; }
  else { leverFill.style.bottom = `${zero + boat.lever / 1.5 * 100}%`; leverFill.style.height = `${-boat.lever / 1.5 * 100}%`; }
  needle.style.left = `${50 - boat.rudder * 45}%`;
  drawCompass(((90 - boat.psi * 180 / Math.PI) % 360 + 360) % 360);
  trailT += dt;
  if (trailT > 0.5) { trailT = 0; trail.push([boat.x, boat.y]); if (trail.length > 160) trail.shift(); }
  drawMinimap();
  if (toastTimer > 0) { toastTimer -= dt; if (toastTimer <= 0) toastEl.classList.remove('show'); }
}

// ---------- Resize ----------
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, isMobile ? 1.25 : 1.5);
  const w = Math.floor(innerWidth * dpr), h = Math.floor(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const hd = Math.min(devicePixelRatio || 1, 2);
  for (const c of [compass, minimap]) {
    const cw = Math.round(c.clientWidth * hd), ch = Math.round(c.clientHeight * hd);
    if (cw && (c.width !== cw || c.height !== ch)) { c.width = cw; c.height = ch; }
  }
}

// ---------- Render ----------
function common(p: Program) {
  return p.v3('uCamPos', cam.eye).v3('uSunDir', SUN_DIR).v3('uSunCol', atm.sun).v3('uAmbTop', atm.ambT).v3('uAmbBot', atm.ambB)
    .v3('uFogCol', atm.fog).f('uFogDen', atm.fogDen).f('uFlash', flash).v3('uFlashDir', flashDir).f('uTime', time);
}
function drawSolid(mesh: { vao: WebGLVertexArrayObject; count: number }, model: Mat4, vp: Mat4, wet = 0) {
  pSolid.m4('uVP', vp).m4('uModel', model).f('uWet', wet);
  G.bindVertexArray(mesh.vao); G.drawArrays(G.TRIANGLES, 0, mesh.count);
}

// Keep a usable horizontal field of view in portrait.
function fitFov(vfov: number, aspect: number) {
  const hmin = 78 * Math.PI / 180;
  return Math.min(Math.max(vfov, 2 * Math.atan(Math.tan(hmin / 2) / aspect)), 105 * Math.PI / 180);
}

function render(boatModel: Mat4, boatInv: Mat4) {
  const aspect = canvas.width / canvas.height;
  cam.fov = fitFov(cam.fov, aspect);
  const view = lookAt(cam.eye, [cam.eye[0] + cam.fwd[0], cam.eye[1] + cam.fwd[1], cam.eye[2] + cam.fwd[2]], cam.up);
  const vp = multiply(perspective(cam.fov, aspect, cam.near, 9000), view);
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.viewport(0, 0, canvas.width, canvas.height);
  G.clearColor(0, 0, 0, 1); G.clear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT);

  // Sky.
  G.disable(G.DEPTH_TEST); G.depthMask(false);
  common(pSky.use()).v3('uFwd', cam.fwd).v3('uRight', cam.right).v3('uUp', cam.up).f('uTanHalf', Math.tan(cam.fov / 2)).f('uAspect', aspect)
    .v3('uSkyTop', atm.sky).v3('uIsland', [IC.x, IC.y, IC.R]).v2('uWindDir', windVec[0], windVec[1]).f('uStorm', smoothstep(0.1, 0.8, atm.s));
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  G.enable(G.DEPTH_TEST); G.depthMask(true);

  // Solids.
  common(pSolid.use());
  drawSolid(land, IDENT, vp, 1);
  BUOYS.forEach((b, i) => drawSolid(buoyMeshes[b.kind], buoyModels[i], vp));
  drawSolid(boatMesh, boatModel, vp);
  const ct = Math.cos(0.6), stt = Math.sin(0.6), spn = boat.rudder * 2.4, cs = Math.cos(spn), ss = Math.sin(spn);
  const local = new Float32Array([ct, 0, -stt, 0, 0, 1, 0, 0, stt, 0, ct, 0, WHEEL_POS[0], WHEEL_POS[1], WHEEL_POS[2], 1]);
  const spin = new Float32Array([1, 0, 0, 0, 0, cs, ss, 0, 0, -ss, cs, 0, 0, 0, 0, 1]);
  drawSolid(wheelMesh, multiply(boatModel, multiply(local, spin)), vp);

  // Ocean.
  common(pOcean.use()).m4('uVP', vp).v2('uCenter', Math.round(cam.eye[0] / 4) * 4, Math.round(cam.eye[1] / 4) * 4)
    .v4a('uWA', uWA).v4a('uWB', uWB).v3('uIsland', [IC.x, IC.y, IC.R]).i('uWake', 0).i('uShore', 1)
    .v3('uWakeRect', [WAKE.ox, WAKE.oy, WAKE.size]).v3('uShoreRect', [IC.x + SHORE.min, IC.y + SHORE.min, SHORE.size])
    .v3('uSkyTop', atm.sky).v2('uWind', windVec[0] * 0.15, windVec[1] * 0.15);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  G.activeTexture(G.TEXTURE1); G.bindTexture(G.TEXTURE_2D, shoreTex);
  G.bindVertexArray(ocean.vao); G.drawElements(G.TRIANGLES, ocean.count, G.UNSIGNED_INT, 0);

  // Translucent and additive passes.
  G.enable(G.BLEND); G.depthMask(false);
  G.blendFunc(G.SRC_ALPHA, G.ONE_MINUS_SRC_ALPHA);
  common(pSpray.use()).m4('uVP', vp).f('uPx', canvas.height / Math.tan(cam.fov / 2));
  G.bindVertexArray(sprayBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, sprayBuf.buf); G.bufferData(G.ARRAY_BUFFER, sprayPos, G.DYNAMIC_DRAW);
  G.drawArrays(G.POINTS, 0, MAXP);

  const onBridge = camMode === 'bridge' && playing;
  const rainCount = Math.floor((isMobile ? 2500 : 6000) * atm.rain);
  if (rainCount > 0) {
    common(pRain.use()).m4('uVP', vp).v3('uCamVel', onBridge ? [boat.vx, boat.vy, boat.vz] : [0, 0, 0])
      .v3('uRainVel', [windVec[0] * 10 * atm.wind, windVec[1] * 10 * atm.wind, -11]).f('uBox', 44).m4('uBoatInv', boatInv)
      .f('uMask', onBridge ? 1 : 0).f('uAmt', 0.4 + 0.6 * atm.rain);
    G.bindVertexArray(rainVao); G.drawArrays(G.LINES, 0, rainCount * 2);
  }

  G.blendFunc(G.ONE, G.ONE);
  const beamAng = time * 0.55;
  const bm = fromAxes([Math.cos(beamAng), Math.sin(beamAng), 0], [-Math.sin(beamAng), Math.cos(beamAng), 0], [0, 0, 1], [LIGHTHOUSE.x, LIGHTHOUSE.y, LIGHTHOUSE.lamp]);
  pBeam.use().m4('uVP', vp).m4('uModel', bm).v3('uCamPos', cam.eye).f('uFogDen', atm.fogDen).f('uI', 0.6 + 0.6 * (1 - atm.s));
  G.bindVertexArray(beam.vao); G.drawArrays(G.TRIANGLES, 0, beam.count);

  // Glows: lighthouse lamp (flares as the beam sweeps past), buoy lights, boat lights.
  glowCount = 0;
  const lamp: V3 = [LIGHTHOUSE.x, LIGHTHOUSE.y, LIGHTHOUSE.lamp];
  const toCam = norm3([cam.eye[0] - lamp[0], cam.eye[1] - lamp[1], 0]);
  const align = Math.abs(Math.cos(beamAng) * toCam[0] + Math.sin(beamAng) * toCam[1]);
  const ld = Math.hypot(cam.eye[0] - lamp[0], cam.eye[1] - lamp[1]);
  const glare = Math.pow(align, 80);
  addGlow(lamp, 5 + ld * 0.004 + glare * (40 + ld * 0.05), [1, 0.85, 0.6], (1.2 + 5 * glare) * Math.exp(-ld * atm.fogDen * 0.45));
  BUOYS.forEach((b, i) => {
    if (((time + b.phase) % 3) >= 0.6) return;
    const d = Math.hypot(cam.eye[0] - buoyTops[i][0], cam.eye[1] - buoyTops[i][1]);
    addGlow(buoyTops[i], 1.2 + d * 0.006, BUOY_LIGHT[b.kind], 1.4 * Math.exp(-d * atm.fogDen * 0.5));
  });
  if (!onBridge) {
    addGlow(boat.local(-0.03, 0, 6.4), 0.8, [1, 1, 0.95], 1.1);
    addGlow(boat.local(0.3, 1.8, 2.4), 0.5, [1, 0.2, 0.2], 1.0);
    addGlow(boat.local(0.3, -1.8, 2.4), 0.5, [0.2, 1, 0.3], 1.0);
  }
  if (glowCount) {
    pGlow.use().m4('uVP', vp).v3('uCamRight', cam.right).v3('uCamUp', cam.up);
    G.bindVertexArray(glowBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, glowBuf.buf);
    G.bufferData(G.ARRAY_BUFFER, glowData.subarray(0, glowCount * 60), G.DYNAMIC_DRAW);
    G.drawArrays(G.TRIANGLES, 0, glowCount * 6);
  }

  // Lightning bolt.
  if (bolt.t < 0.32 && bolt.pts.length) {
    const verts: number[] = [];
    bolt.pts.forEach((line, bi) => {
      for (let k = 0; k < line.length - 1; k++) {
        const p = line[k], q = line[k + 1];
        const mid: V3 = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
        const w = (bi === 0 ? 9 : 5) + Math.hypot(mid[0] - cam.eye[0], mid[1] - cam.eye[1]) * 0.003;
        const sd = norm3(cross(sub(q, p), sub(cam.eye, mid)));
        const a0: V3 = [p[0] - sd[0] * w, p[1] - sd[1] * w, p[2] - sd[2] * w], a1: V3 = [p[0] + sd[0] * w, p[1] + sd[1] * w, p[2] + sd[2] * w];
        const b0: V3 = [q[0] - sd[0] * w, q[1] - sd[1] * w, q[2] - sd[2] * w], b1: V3 = [q[0] + sd[0] * w, q[1] + sd[1] * w, q[2] + sd[2] * w];
        verts.push(...a0, 0, ...a1, 1, ...b1, 1, ...a0, 0, ...b1, 1, ...b0, 0);
      }
    });
    pBolt.use().m4('uVP', vp).f('uI', flash * 2.2);
    G.bindVertexArray(boltBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, boltBuf.buf);
    G.bufferData(G.ARRAY_BUFFER, new Float32Array(verts), G.DYNAMIC_DRAW);
    G.drawArrays(G.TRIANGLES, 0, verts.length / 4);
  }
  G.disable(G.BLEND); G.depthMask(true);
  G.bindVertexArray(null);
}

// ---------- Game step ----------
function handleEvents() {
  for (const e of boat.events) {
    if (e.type === 'slam') {
      audio.slap(e.i);
      cam.shake = Math.max(cam.shake, 0.08 * e.i);
      wakeBlob = 0.6 * e.i;
      const n = Math.round(30 + 80 * e.i);
      for (let k = 0; k < n; k++) emitLocal([5.6 + R() * 1.4, (R() - 0.5) * 3.4, 1.4 + R() * 0.6], [-(1 + R() * 5 * e.i), (R() - 0.5) * (4 + 8 * e.i), 3 + R() * (5 + 9 * e.i)]);
    } else if (e.type === 'green') {
      for (let k = 0; k < 70 * e.i; k++) emitLocal([3 + R() * 3, (R() - 0.5) * 3, 1.8 + R()], [-(4 + R() * 5), (R() - 0.5) * 3, 2 + R() * 3]);
      cam.shake = Math.max(cam.shake, 0.05);
    } else {
      audio.crunch(e.i);
      cam.shake = Math.max(cam.shake, 0.25 * e.i);
      toast('Rocks!', 1.5);
    }
  }
  boat.events.length = 0;
}

function step(dt: number) {
  if (playing) updateControls(dt);
  const n = Math.max(1, Math.ceil(dt / (1 / 60)));
  for (let i = 0; i < n; i++) boat.step(dt / n, windVec[0] * 14 * atm.wind, windVec[1] * 14 * atm.wind);
  handleEvents();
  const speed = Math.hypot(boat.vx, boat.vy);
  if (speed > 3.5 && boat.imm > -0.9) {
    const k = speed > 8 ? 2 : 1;
    for (let j = 0; j < k; j++) for (const sg of [1, -1]) emitLocal([5.2 + R(), sg * 1.7, 0.6], [-1 - R(), sg * (1.5 + R() * 2.5), 1.2 + R() * 2.2], 0.7);
  }
  if (playing) checkGates();
}

function simulate(dt: number) {
  time += dt;
  updatePhases(time);
  step(dt);
  updateAtmosphere(dt);
  updateLightning(dt);
  updateBuoys();
  const ax = boatAxes(boat.psi, boat.th, boat.ph);
  const pos: V3 = [boat.x, boat.y, boat.z];
  const model = fromAxes(ax.X, ax.Y, ax.Z, pos), inv = rigidInverse(ax.X, ax.Y, ax.Z, pos);
  updateSpray(dt, inv);
  updateWake();
  updateCamera(dt);
  return { model, inv };
}

// ---------- Main loop ----------
let last = performance.now();
function frame(now: number) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  resize();
  const { model, inv } = simulate(dt);
  render(model, inv);
  updateHud(dt);
  audio.update({
    speed: Math.min(Math.hypot(boat.vx, boat.vy) / 11, 1), rpm: boat.rpm, storm: 0.15 + 0.85 * atm.s, rain: atm.rain, wind: atm.wind,
    heave: Math.abs(boat.vz), roll: Math.abs(boat.phv),
  }, dt);
  requestAnimationFrame(frame);
}

$('go').addEventListener('click', () => {
  playing = true;
  document.body.classList.add('playing');
  audio.start();
  boat.lever = 0;
  resetWakeMarks();
  if (isTouch) { document.body.classList.add('hint'); setTimeout(() => document.body.classList.remove('hint'), 6000); }
  setTimeout(() => toast('Sheltered water'), 1800);
});
$('camBtn').addEventListener('click', toggleCam);
$('muteBtn').addEventListener('click', toggleMute);
$('fsBtn').addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen?.().catch(() => undefined);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) last = performance.now(); });

updateAtmosphere(10);
requestAnimationFrame(frame);

// Dev hooks: deterministic stepping while the tab is throttled, and state inspection.
const dev = window as unknown as Record<string, unknown>;
dev.__advance = (n: number, dt = 1 / 60) => {
  let r = simulate(dt);
  for (let i = 1; i < n; i++) r = simulate(dt);
  render(r.model, r.inv); updateHud(0);
};
dev.__game = {
  boat, keys, cam, atm, start: () => $('go').click(), toggleCam,
  teleport: (x: number, y: number, psi: number) => { boat.x = x; boat.y = y; boat.psi = psi; boat.vx = boat.vy = 0; resetWakeMarks(); gatePrev = [x, y]; },
};
