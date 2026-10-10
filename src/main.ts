import {
  VERT, ADVECT, SPLAT, DIVERGENCE, JACOBI, GRADIENT_SUBTRACT, BOUNDARY, DISPLAY, WAVE, VERT3D, FRAG3D,
  BOAT_STATE, VERT_BOAT, FRAG_BOAT,
} from './shaders';
import { Mat4, perspective, lookAt, multiply, ortho } from './mat4';

// ---------- WebGL setup ----------
const canvas = document.getElementById('c') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'high-performance' });
if (!gl) throw new Error('WebGL2 is required');
const G: WebGL2RenderingContext = gl;

// Pick a renderable float texture format. RGBA16F is the most portable; some
// desktops only expose EXT_color_buffer_float (32F), some phones only half-float.
G.getExtension('EXT_color_buffer_float');
G.getExtension('EXT_color_buffer_half_float');
const linearFloat = !!G.getExtension('OES_texture_float_linear');

interface TexFormat { internal: number; format: number; type: number; filter: number; }
function probeFormat(internal: number, type: number, filter: number): TexFormat | null {
  const tex = G.createTexture()!;
  G.bindTexture(G.TEXTURE_2D, tex);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, filter);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, filter);
  G.texImage2D(G.TEXTURE_2D, 0, internal, 4, 4, 0, G.RGBA, type, null);
  const fb = G.createFramebuffer()!;
  G.bindFramebuffer(G.FRAMEBUFFER, fb);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT0, G.TEXTURE_2D, tex, 0);
  const ok = G.checkFramebufferStatus(G.FRAMEBUFFER) === G.FRAMEBUFFER_COMPLETE;
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.deleteFramebuffer(fb); G.deleteTexture(tex);
  return ok ? { internal, format: G.RGBA, type, filter } : null;
}
const probed =
  probeFormat(G.RGBA16F, G.HALF_FLOAT, G.LINEAR) ??
  probeFormat(G.RGBA32F, G.FLOAT, linearFloat ? G.LINEAR : G.NEAREST);
if (!probed) throw new Error('Float render targets are not supported on this device');
const FMT: TexFormat = probed;

const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || navigator.maxTouchPoints > 1;
const SIM_RES = isMobile ? 160 : 256;
const DYE_RES = isMobile ? 384 : 768;
const MESH_RES = isMobile ? 192 : 320;
const PRESSURE_ITERS = isMobile ? 24 : 40;
const VISC_ITERS = isMobile ? 12 : 20;

// ---------- Shader helpers ----------
function compile(type: number, src: string): WebGLShader {
  const s = G.createShader(type)!;
  G.shaderSource(s, src);
  G.compileShader(s);
  if (!G.getShaderParameter(s, G.COMPILE_STATUS)) {
    throw new Error(G.getShaderInfoLog(s) ?? 'shader compile error');
  }
  return s;
}

class Program {
  prog: WebGLProgram;
  uniforms = new Map<string, WebGLUniformLocation>();
  constructor(frag: string, vert: string = VERT, attribs: string[] = ['aPos']) {
    const p = G.createProgram()!;
    G.attachShader(p, compile(G.VERTEX_SHADER, vert));
    G.attachShader(p, compile(G.FRAGMENT_SHADER, frag));
    attribs.forEach((name, i) => G.bindAttribLocation(p, i, name));
    G.linkProgram(p);
    if (!G.getProgramParameter(p, G.LINK_STATUS)) throw new Error(G.getProgramInfoLog(p) ?? 'link error');
    this.prog = p;
    const n = G.getProgramParameter(p, G.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < n; i++) {
      const info = G.getActiveUniform(p, i)!;
      this.uniforms.set(info.name, G.getUniformLocation(p, info.name)!);
    }
  }
  use() { G.useProgram(this.prog); return this; }
  f(name: string, v: number) { G.uniform1f(this.uniforms.get(name)!, v); return this; }
  v2(name: string, x: number, y: number) { G.uniform2f(this.uniforms.get(name)!, x, y); return this; }
  v3(name: string, x: number, y: number, z: number) { G.uniform3f(this.uniforms.get(name)!, x, y, z); return this; }
  m4(name: string, m: Mat4) { G.uniformMatrix4fv(this.uniforms.get(name)!, false, m); return this; }
  tex(name: string, unit: number) { G.uniform1i(this.uniforms.get(name)!, unit); return this; }
}

// Fullscreen quad
const quadVao = G.createVertexArray()!;
G.bindVertexArray(quadVao);
const vbo = G.createBuffer()!;
G.bindBuffer(G.ARRAY_BUFFER, vbo);
G.bufferData(G.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), G.STATIC_DRAW);
G.enableVertexAttribArray(0);
G.vertexAttribPointer(0, 2, G.FLOAT, false, 0, 0);

// Surface mesh for 3D view: (MESH_RES+1)^2 grid of uv vertices.
const meshVao = G.createVertexArray()!;
let meshIndexCount = 0;
function buildMesh(nx: number, ny: number) {
  G.bindVertexArray(meshVao);
  const verts = new Float32Array((nx + 1) * (ny + 1) * 2);
  let k = 0;
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) { verts[k++] = i / nx; verts[k++] = j / ny; }
  const idx = new Uint32Array(nx * ny * 6);
  k = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    idx[k++] = a; idx[k++] = b; idx[k++] = c; idx[k++] = b; idx[k++] = d; idx[k++] = c;
  }
  const vb = G.createBuffer()!;
  G.bindBuffer(G.ARRAY_BUFFER, vb);
  G.bufferData(G.ARRAY_BUFFER, verts, G.STATIC_DRAW);
  G.enableVertexAttribArray(0);
  G.vertexAttribPointer(0, 2, G.FLOAT, false, 0, 0);
  const ib = G.createBuffer()!;
  G.bindBuffer(G.ELEMENT_ARRAY_BUFFER, ib);
  G.bufferData(G.ELEMENT_ARRAY_BUFFER, idx, G.STATIC_DRAW);
  meshIndexCount = idx.length;
  G.bindVertexArray(quadVao);
}

// Boat: low-poly hull + deck + cabin, flat shaded. Local frame: x forward, y starboard, z up.
const boatVao = G.createVertexArray()!;
let boatVertCount = 0;
function buildBoat() {
  const v: number[] = [];
  type P = [number, number, number];
  const tri = (a: P, b: P, c: P, col: P) => { for (const q of [a, b, c]) v.push(...q, ...col); };
  const quad = (a: P, b: P, c: P, d: P, col: P) => { tri(a, b, c, col); tri(a, c, d, col); };
  const HULL: P = [0.62, 0.16, 0.12], DECK: P = [0.85, 0.70, 0.45], CABIN: P = [0.95, 0.95, 0.92], ROOF: P = [0.35, 0.38, 0.42];
  const N = 14, G0 = 0.14; // gunwale height
  const sect = (i: number) => {
    const t = i / N;                           // 0 = stern, 1 = bow
    const x = -0.5 + t;
    const w = 0.30 * Math.pow(Math.sin(Math.PI * (0.12 + 0.88 * t * 0.92)), 0.7) * (1 - 0.25 * t * t);
    const k = 0.14 * (0.6 + 0.4 * Math.sin(Math.PI * t)) * (1 - 0.6 * t * t * t);
    return { L: [x, -w, G0] as P, R: [x, w, G0] as P, K: [x, 0, -k] as P };
  };
  const secs = Array.from({ length: N + 1 }, (_, i) => sect(i));
  for (let i = 0; i < N; i++) {
    const a = secs[i], b = secs[i + 1];
    quad(a.L, a.K, b.K, b.L, HULL);
    quad(a.R, b.R, b.K, a.K, HULL);
    quad(a.L, b.L, b.R, a.R, DECK);
  }
  const st = secs[0]; tri(st.L, st.R, st.K, HULL); // transom
  // Cabin box
  const x0 = -0.22, x1 = 0.08, y = 0.15, z0 = G0, z1 = 0.34;
  quad([x0, -y, z0], [x1, -y, z0], [x1, -y, z1], [x0, -y, z1], CABIN);
  quad([x0, y, z0], [x0, y, z1], [x1, y, z1], [x1, y, z0], CABIN);
  quad([x1, -y, z0], [x1, y, z0], [x1, y, z1], [x1, -y, z1], CABIN);
  quad([x0, -y, z0], [x0, -y, z1], [x0, y, z1], [x0, y, z0], CABIN);
  quad([x0, -y, z1], [x1, -y, z1], [x1, y, z1], [x0, y, z1], ROOF);
  G.bindVertexArray(boatVao);
  const buf = G.createBuffer()!;
  G.bindBuffer(G.ARRAY_BUFFER, buf);
  G.bufferData(G.ARRAY_BUFFER, new Float32Array(v), G.STATIC_DRAW);
  G.enableVertexAttribArray(0); G.vertexAttribPointer(0, 3, G.FLOAT, false, 24, 0);
  G.enableVertexAttribArray(1); G.vertexAttribPointer(1, 3, G.FLOAT, false, 24, 12);
  boatVertCount = v.length / 6;
  G.bindVertexArray(quadVao);
}
buildBoat();

// ---------- Framebuffers ----------
interface FBO { tex: WebGLTexture; fb: WebGLFramebuffer; w: number; h: number; attach(unit: number): number; }
interface DoubleFBO { read: FBO; write: FBO; w: number; h: number; swap(): void; }

function createFBO(w: number, h: number): FBO {
  const tex = G.createTexture()!;
  G.activeTexture(G.TEXTURE0);
  G.bindTexture(G.TEXTURE_2D, tex);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, FMT.filter);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, FMT.filter);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.CLAMP_TO_EDGE);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.CLAMP_TO_EDGE);
  G.texImage2D(G.TEXTURE_2D, 0, FMT.internal, w, h, 0, FMT.format, FMT.type, null);
  const fb = G.createFramebuffer()!;
  G.bindFramebuffer(G.FRAMEBUFFER, fb);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT0, G.TEXTURE_2D, tex, 0);
  if (G.checkFramebufferStatus(G.FRAMEBUFFER) !== G.FRAMEBUFFER_COMPLETE) throw new Error('FBO incomplete');
  G.viewport(0, 0, w, h);
  G.clearColor(0, 0, 0, 0);
  G.clear(G.COLOR_BUFFER_BIT);
  return {
    tex, fb, w, h,
    attach(unit: number) { G.activeTexture(G.TEXTURE0 + unit); G.bindTexture(G.TEXTURE_2D, tex); return unit; },
  };
}
function createDoubleFBO(w: number, h: number): DoubleFBO {
  let a = createFBO(w, h);
  let b = createFBO(w, h);
  return {
    w, h,
    get read() { return a; }, get write() { return b; },
    swap() { const t = a; a = b; b = t; },
  };
}

function blit(target: FBO | null) {
  G.bindVertexArray(quadVao);
  if (target) { G.bindFramebuffer(G.FRAMEBUFFER, target.fb); G.viewport(0, 0, target.w, target.h); }
  else { G.bindFramebuffer(G.FRAMEBUFFER, null); G.viewport(0, 0, canvas.width, canvas.height); }
  G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
}

// ---------- Programs ----------
const pAdvect = new Program(ADVECT);
const pSplat = new Program(SPLAT);
const pDiv = new Program(DIVERGENCE);
const pJacobi = new Program(JACOBI);
const pGrad = new Program(GRADIENT_SUBTRACT);
const pBound = new Program(BOUNDARY);
const pWave = new Program(WAVE);
const pDisplay = new Program(DISPLAY);
const pMesh = new Program(FRAG3D, VERT3D, ['aUv']);
const pBoatState = new Program(BOAT_STATE);
const pBoat = new Program(FRAG_BOAT, VERT_BOAT, ['aPos', 'aCol']);

// ---------- State ----------
// dye: 2D view's surface shading field (original look). wave: 3D view's height field (x = h, y = dh/dt).
let velocity!: DoubleFBO, pressure!: DoubleFBO, divergence!: FBO, viscB!: FBO, dye!: DoubleFBO, wave!: DoubleFBO;
const boatState = createDoubleFBO(1, 1);
const BOAT_SCALE = 0.26;
let simW = 0, simH = 0, dyeW = 0, dyeH = 0;
let aspect = 1;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (w === 0 || h === 0) return; // not laid out yet; try again next frame
  if (canvas.width === w && canvas.height === h && velocity) return;
  canvas.width = w; canvas.height = h;
  aspect = w / h;
  const dims = (res: number) => aspect >= 1
    ? [Math.round(res * aspect), res] : [res, Math.round(res / aspect)];
  [simW, simH] = dims(SIM_RES);
  [dyeW, dyeH] = dims(DYE_RES);
  velocity = createDoubleFBO(simW, simH);
  pressure = createDoubleFBO(simW, simH);
  divergence = createFBO(simW, simH);
  viscB = createFBO(simW, simH);
  dye = createDoubleFBO(dyeW, dyeH);
  wave = createDoubleFBO(dyeW, dyeH);
  const [mx, my] = dims(MESH_RES);
  buildMesh(mx, my);
}
window.addEventListener('resize', resize);
resize();

// ---------- UI ----------
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const viscSlider = $<HTMLInputElement>('visc');
const viscOut = $<HTMLOutputElement>('viscOut');
const splatSlider = $<HTMLInputElement>('splat');
const heightSlider = $<HTMLInputElement>('height');
const speedSlider = $<HTMLInputElement>('speed');
const speedOut = $<HTMLOutputElement>('speedOut');
const boatBtn = $<HTMLButtonElement>('boat');
let showBoat = true;
boatBtn.addEventListener('click', () => { showBoat = !showBoat; boatBtn.classList.toggle('active', showBoat); });
function speedFactor() { return Number(speedSlider.value) / 100; }
function updateSpeedLabel() { speedOut.value = speedFactor().toFixed(1) + 'x'; }
speedSlider.addEventListener('input', updateSpeedLabel);
updateSpeedLabel();
const dampButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#damp button'));
const viewButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#view button'));
const hint = $<HTMLDivElement>('hint');
// [velocity dissipation, wave damping]
const DAMP_LEVELS: [number, number][] = [[1.0, 0.998], [0.995, 0.99], [0.975, 0.97], [0.93, 0.92]];
let dampLevel = 1;
let paused = false;
let view3D = false;

function viscosity(): number {
  const t = Number(viscSlider.value) / 100;
  return 3 * t * t;
}
function updateViscLabel() { viscOut.value = viscosity().toFixed(3); }
viscSlider.addEventListener('input', updateViscLabel);
updateViscLabel();

dampButtons.forEach((b) => b.addEventListener('click', () => {
  dampLevel = Number(b.dataset.damp);
  dampButtons.forEach((x) => x.classList.toggle('active', x === b));
}));
viewButtons.forEach((b) => b.addEventListener('click', () => {
  view3D = b.dataset.view === '3d';
  viewButtons.forEach((x) => x.classList.toggle('active', x === b));
  document.body.classList.toggle('view3d', view3D);
  hint.textContent = view3D
    ? 'Drag to make waves. Two fingers / right-drag to orbit, pinch or wheel to zoom.'
    : 'Drag with the mouse (or touch) to push the water. Walls reflect.';
}));
function clearAll() {
  for (const f of [velocity.read, velocity.write, pressure.read, pressure.write, divergence, viscB, dye.read, dye.write,
    wave.read, wave.write, boatState.read, boatState.write]) {
    G.bindFramebuffer(G.FRAMEBUFFER, f.fb); G.clearColor(0, 0, 0, 0); G.clear(G.COLOR_BUFFER_BIT);
  }
}
$('reset').addEventListener('click', clearAll);
const pauseBtn = $<HTMLButtonElement>('pause');
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; });

// ---------- Camera (3D view) ----------
const cam = { yaw: 0.0, pitch: 0.95, dist: 3.2 };
function cameraEye(): [number, number, number] {
  const cp = Math.cos(cam.pitch);
  return [cam.dist * cp * Math.sin(cam.yaw), -cam.dist * cp * Math.cos(cam.yaw), cam.dist * Math.sin(cam.pitch)];
}
const FOV = 45 * Math.PI / 180;
function viewProj(): Mat4 {
  const eye = cameraEye();
  return multiply(perspective(FOV, aspect, 0.05, 50), lookAt(eye, [0, 0, 0], [0, 0, 1]));
}
// Screen (0..1, y down) -> sim uv by intersecting the camera ray with the z = 0 plane.
function unproject(sx: number, sy: number): [number, number] | null {
  const eye = cameraEye();
  const fx = -eye[0], fy = -eye[1], fz = -eye[2];
  const fl = Math.hypot(fx, fy, fz);
  const f = [fx / fl, fy / fl, fz / fl];
  // right = f x up(0,0,1); up' = right x f
  let r = [f[1], -f[0], 0]; const rl = Math.hypot(r[0], r[1]); r = [r[0] / rl, r[1] / rl, 0];
  const u = [r[1] * f[2] - 0 * f[1], 0 * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  const th = Math.tan(FOV / 2);
  const nx = (sx * 2 - 1) * th * aspect, ny = (1 - sy * 2) * th;
  const d = [f[0] + nx * r[0] + ny * u[0], f[1] + nx * r[1] + ny * u[1], f[2] + nx * r[2] + ny * u[2]];
  if (d[2] >= -1e-5) return null;
  const t = -eye[2] / d[2];
  const px = eye[0] + d[0] * t, py = eye[1] + d[1] * t;
  return [px / (2 * aspect) + 0.5, py / 2 + 0.5];
}

// ---------- Pointer input (mouse + touch via Pointer Events) ----------
interface Pointer { x: number; y: number; dx: number; dy: number; moved: boolean; sx: number; sy: number; orbit: boolean; }
const pointers = new Map<number, Pointer>();
let pinchDist = 0;

function screenPos(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}
function toSim(sx: number, sy: number): [number, number] | null {
  return view3D ? unproject(sx, sy) : [sx, 1 - sy];
}
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  const [sx, sy] = screenPos(e);
  const p = toSim(sx, sy) ?? [0, 0];
  const orbit = view3D && (e.button === 2 || e.shiftKey);
  pointers.set(e.pointerId, { x: p[0], y: p[1], dx: 0, dy: 0, moved: false, sx, sy, orbit });
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDist = Math.hypot(a.sx - b.sx, a.sy - b.sy);
  }
});
canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const [sx, sy] = screenPos(e);
  const multiTouch = view3D && pointers.size >= 2;
  if (p.orbit || multiTouch) {
    // Orbit: in multi-touch, each finger contributes half; pinch adjusts distance.
    const scale = multiTouch ? 0.5 : 1;
    cam.yaw += (sx - p.sx) * 3.0 * scale;
    cam.pitch = Math.min(1.5, Math.max(0.15, cam.pitch + (sy - p.sy) * 2.0 * scale));
    p.sx = sx; p.sy = sy;
    if (multiTouch) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.sx - b.sx, a.sy - b.sy);
      if (pinchDist > 0) cam.dist = Math.min(8, Math.max(1.2, cam.dist * (pinchDist / d)));
      pinchDist = d;
    }
    return;
  }
  const s = toSim(sx, sy);
  p.sx = sx; p.sy = sy;
  if (!s) return;
  p.dx += s[0] - p.x; p.dy += s[1] - p.y; p.x = s[0]; p.y = s[1]; p.moved = true;
});
const end = (e: PointerEvent) => { pointers.delete(e.pointerId); pinchDist = 0; };
canvas.addEventListener('pointerup', end);
canvas.addEventListener('pointercancel', end);
canvas.addEventListener('pointerleave', end);
canvas.addEventListener('wheel', (e) => {
  if (!view3D) return;
  e.preventDefault();
  cam.dist = Math.min(8, Math.max(1.2, cam.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

// ---------- Simulation step ----------
function splat(x: number, y: number, dx: number, dy: number) {
  const radius = (Number(splatSlider.value) / 100) * 0.01 + 0.0004;
  const force = 1400;
  const speed = Math.hypot(dx, dy);

  pSplat.use().v2('uTexel', 1 / simW, 1 / simH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
    .tex('uTarget', velocity.read.attach(0)).v3('uValue', dx * force, dy * force, 0);
  blit(velocity.write); velocity.swap();

  if (view3D) {
    // Push the surface down where the finger moves; the wave equation turns it into ripples.
    pSplat.use().v2('uTexel', 1 / dyeW, 1 / dyeH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
      .tex('uTarget', wave.read.attach(0)).v3('uValue', -Math.min(speed * 8, 0.3), 0, 0);
    blit(wave.write); wave.swap();
  } else {
    // 2D: brighten the stirred water.
    pSplat.use().v2('uTexel', 1 / dyeW, 1 / dyeH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
      .tex('uTarget', dye.read.attach(0)).v3('uValue', Math.min(speed * 40, 1.2), 0, 0);
    blit(dye.write); dye.swap();
  }
}

function step(dt0: number) {
  const tx = 1 / simW, ty = 1 / simH;
  const sf = speedFactor();
  const dt = dt0 * sf; // fluid time step
  const [velDamp, waveDamp] = DAMP_LEVELS[dampLevel];

  for (const p of pointers.values()) {
    if (p.moved && (p.dx !== 0 || p.dy !== 0) && !p.orbit) {
      // Clamp per-frame displacement so unprojection jumps near the horizon can't inject huge forces.
      const m = Math.hypot(p.dx, p.dy), cap = 0.04;
      const k = m > cap ? cap / m : 1;
      splat(p.x, p.y, p.dx * k, p.dy * k);
    }
    p.dx = 0; p.dy = 0; p.moved = false;
  }

  // 1. Advect velocity (self-advection) with damping.
  pAdvect.use().v2('uTexel', tx, ty).f('uDt', dt).f('uDissipation', velDamp)
    .tex('uVelocity', velocity.read.attach(0)).tex('uSource', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();

  // 2. Viscous diffusion: (I - nu*dt*Lap) v = v0, solved with Jacobi.
  const nu = viscosity();
  if (nu > 0) {
    const alpha = 1 / (nu * dt);
    const rBeta = 1 / (4 + alpha);
    pAdvect.use().v2('uTexel', tx, ty).f('uDt', 0).f('uDissipation', 1)
      .tex('uVelocity', velocity.read.attach(0)).tex('uSource', velocity.read.attach(0));
    blit(viscB);
    pJacobi.use().v2('uTexel', tx, ty).f('uAlpha', alpha).f('uRBeta', rBeta).tex('uB', viscB.attach(1));
    for (let i = 0; i < VISC_ITERS; i++) {
      pJacobi.tex('uX', velocity.read.attach(0));
      blit(velocity.write); velocity.swap();
    }
  }

  // 3. Reflecting boundary.
  pBound.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();

  // 4. Divergence.
  pDiv.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(divergence);

  // 5. Pressure Poisson: Lap p = div.
  pJacobi.use().v2('uTexel', tx, ty).f('uAlpha', -1).f('uRBeta', 0.25).tex('uB', divergence.attach(1));
  for (let i = 0; i < PRESSURE_ITERS; i++) {
    pJacobi.tex('uX', pressure.read.attach(0));
    blit(pressure.write); pressure.swap();
  }

  // 6. Project: v -= grad p.
  pGrad.use().v2('uTexel', tx, ty).tex('uPressure', pressure.read.attach(0)).tex('uVelocity', velocity.read.attach(1));
  blit(velocity.write); velocity.swap();

  // 7. Boundary again after projection.
  pBound.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();

  // 8. Boat: drift with the current, turn toward its motion.
  pBoatState.use().v2('uTexel', tx, ty).f('uDt', dt).f('uAspect', aspect)
    .tex('uState', boatState.read.attach(0)).tex('uVelocity', velocity.read.attach(1));
  blit(boatState.write); boatState.swap();

  if (!view3D) {
    // 9 (2D). Advect the shading field with the flow and let it fade.
    pAdvect.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', dt * (dyeW / simW)).f('uDissipation', 0.985)
      .tex('uVelocity', velocity.read.attach(0)).tex('uSource', dye.read.attach(1));
    blit(dye.write); dye.swap();
    return;
  }

  // 9 (3D). Advect the height field with the flow.
  pAdvect.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', dt * (dyeW / simW)).f('uDissipation', 1.0)
    .tex('uVelocity', velocity.read.attach(0)).tex('uSource', wave.read.attach(1));
  blit(wave.write); wave.swap();

  // 10 (3D). Surface waves: h_tt = c^2 Lap h (explicit, CFL: c^2 dt^2 <= 0.5). Sub-stepped so the
  // speed slider can raise the wave speed beyond the CFL limit of one step per frame.
  const subs = Math.max(1, Math.round(2 * sf));
  const wdt = dt0;
  const c2 = Math.min(0.3, 0.4 / Math.max(wdt * wdt, 1e-3));
  const wd = Math.pow(waveDamp * (1 - 0.01 * nu), 1 / subs); // viscosity also damps the surface a little
  pWave.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', wdt).f('uC2', c2).f('uDamp', wd)
    .f('uAspect', aspect).f('uBoatSize', showBoat ? BOAT_SCALE * 0.25 : 0.0001)
    .tex('uBoat', boatState.read.attach(1));
  for (let i = 0; i < subs; i++) {
    pWave.tex('uField', wave.read.attach(0));
    blit(wave.write); wave.swap();
  }
}

function drawBoat(vp: Mat4, eye: [number, number, number], heightScale: number) {
  if (!showBoat) return;
  pBoat.use().m4('uVP', vp).v2('uTexel', 1 / dyeW, 1 / dyeH).f('uAspect', aspect).f('uHeightScale', heightScale)
    .f('uScale', BOAT_SCALE).v3('uEye', eye[0], eye[1], eye[2])
    .tex('uState', boatState.read.attach(0)).tex('uHeight', wave.read.attach(1));
  G.bindVertexArray(boatVao);
  G.drawArrays(G.TRIANGLES, 0, boatVertCount);
  G.bindVertexArray(quadVao);
}

function render2D(t: number) {
  pDisplay.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uTime', t)
    .tex('uDye', dye.read.attach(0)).tex('uPressure', pressure.read.attach(1)).tex('uVelocity', velocity.read.attach(2));
  blit(null);
  // Top-down boat: orthographic camera straight above the pool.
  G.enable(G.DEPTH_TEST);
  G.clear(G.DEPTH_BUFFER_BIT);
  drawBoat(ortho(-aspect, aspect, -1, 1, -10, 10), [0, 0, 10], 0.0);
  G.disable(G.DEPTH_TEST);
}

function render3D() {
  const eye = cameraEye();
  const heightScale = 0.06 + (Number(heightSlider.value) / 100) * 0.5;
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.viewport(0, 0, canvas.width, canvas.height);
  G.clearColor(0.03, 0.07, 0.12, 1);
  G.enable(G.DEPTH_TEST);
  G.clear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT);
  pMesh.use().m4('uVP', viewProj()).f('uAspect', aspect).f('uHeightScale', heightScale)
    .v2('uTexel', 1 / dyeW, 1 / dyeH).v3('uEye', eye[0], eye[1], eye[2])
    .tex('uHeight', wave.read.attach(0)).tex('uVelocity', velocity.read.attach(1));
  G.bindVertexArray(meshVao);
  G.drawElements(G.TRIANGLES, meshIndexCount, G.UNSIGNED_INT, 0);
  G.bindVertexArray(quadVao);
  drawBoat(viewProj(), eye, heightScale);
  G.disable(G.DEPTH_TEST);
}

let last = performance.now();
function frame(now: number) {
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  resize();
  if (!velocity) { requestAnimationFrame(frame); return; }
  if (!paused) step(dt * 60); // dt in "frames" so velocity is in texels/frame
  if (view3D) render3D(); else render2D(now / 1000);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Debug readback (dev only): sample the sim textures.
(window as unknown as { __dbg: () => unknown }).__dbg = () => {
  const rd = (f: FBO, x: number, y: number) => {
    G.bindFramebuffer(G.FRAMEBUFFER, f.fb);
    const type = G.getParameter(G.IMPLEMENTATION_COLOR_READ_TYPE) as number;
    const fmt = G.getParameter(G.IMPLEMENTATION_COLOR_READ_FORMAT) as number;
    const out = type === G.FLOAT ? new Float32Array(4) : new Uint16Array(4);
    G.readPixels(x, y, 1, 1, fmt, type, out);
    return { type, fmt, v: Array.from(out) };
  };
  return {
    boat: rd(boatState.read, 0, 0),
    velC: rd(velocity.read, simW >> 1, simH >> 1),
    dyeC: rd(dye.read, dyeW >> 1, dyeH >> 1), waveC: rd(wave.read, dyeW >> 1, dyeH >> 1),
    dye0: rd(dye.read, 2, 2),
    cam: { ...cam }, aspect, up: unproject(0.5, 0.6),
  };
};
