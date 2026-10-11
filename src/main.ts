import {
  VERT, ADVECT, SPLAT, DIVERGENCE, JACOBI, GRADIENT_SUBTRACT, BOUNDARY, DISPLAY, WAVE, VERT3D, FRAG3D,
  VERT_BOAT, FRAG_BOAT, PARTICLE_UPDATE, VERT_PARTICLE, FRAG_PARTICLE,
  BALL_PUSH, BALL_VEL, BOAT_INFO, SHIFT, SKY_PASS,
} from './shaders';
import { Mat4, perspective, lookAt, multiply, ortho } from './mat4';
import { SeaAudio } from './audio';

// ---------- WebGL setup ----------
const canvas = document.getElementById('c') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2', { antialias: new URLSearchParams(location.search).get('aa') === '1', alpha: false, powerPreference: 'high-performance' });
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
const HALF = FMT.type === G.HALF_FLOAT;

// Float <-> half-float helpers for uploading / reading small state textures.
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toHalf(v: number): number {
  f32[0] = v; const x = u32[0];
  const sign = (x >> 16) & 0x8000; let exp = ((x >> 23) & 0xff) - 127 + 15; let mant = x & 0x7fffff;
  if (exp <= 0) return sign; if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | (mant >> 13);
}
function fromHalf(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || navigator.maxTouchPoints > 1;
if (navigator.maxTouchPoints > 0) document.body.classList.add('touch');
const Q = new URLSearchParams(location.search);
const qn = (k: string, d: number) => { const v = Q.get(k); return v === null ? d : Number(v); };
const SIM_RES = qn('sim', isMobile ? 160 : 256);
const DYE_RES = qn('dye', isMobile ? 384 : 640);
const MESH_RES = qn('mesh', isMobile ? 160 : 256);
const PRESSURE_ITERS = qn('pi', isMobile ? 20 : 30);
const VISC_ITERS = qn('vi', isMobile ? 8 : 12);
const MAX_DPR = qn('dpr', 1.5);

// ---------- Shader helpers ----------
function compile(type: number, src: string): WebGLShader {
  const s = G.createShader(type)!;
  G.shaderSource(s, src);
  G.compileShader(s);
  if (!G.getShaderParameter(s, G.COMPILE_STATUS)) throw new Error(G.getShaderInfoLog(s) ?? 'shader compile error');
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
  f(name: string, v: number) { const u = this.uniforms.get(name); if (u) G.uniform1f(u, v); return this; }
  i(name: string, v: number) { const u = this.uniforms.get(name); if (u) G.uniform1i(u, v); return this; }
  v2(name: string, x: number, y: number) { const u = this.uniforms.get(name); if (u) G.uniform2f(u, x, y); return this; }
  v3(name: string, x: number, y: number, z: number) { const u = this.uniforms.get(name); if (u) G.uniform3f(u, x, y, z); return this; }
  m4(name: string, m: Mat4) { G.uniformMatrix4fv(this.uniforms.get(name)!, false, m); return this; }
  tex(name: string, unit: number) { const u = this.uniforms.get(name); if (u) G.uniform1i(u, unit); return this; }
}

// Fullscreen quad
const quadVao = G.createVertexArray()!;
G.bindVertexArray(quadVao);
const vbo = G.createBuffer()!;
G.bindBuffer(G.ARRAY_BUFFER, vbo);
G.bufferData(G.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), G.STATIC_DRAW);
G.enableVertexAttribArray(0);
G.vertexAttribPointer(0, 2, G.FLOAT, false, 0, 0);

// Ocean mesh: a grid in uv space that covers [-3, 4] with density concentrated on the
// simulation window [0, 1]; outside the window only the analytic swell is rendered.
const meshVao = G.createVertexArray()!;
let meshIndexCount = 0;
function buildMesh(nx: number, ny: number) {
  G.bindVertexArray(meshVao);
  const warp = (t: number) => { const x = 2 * t - 1; return 0.5 + x * (0.5 + 3.0 * x * x * x * x); };
  const verts = new Float32Array((nx + 1) * (ny + 1) * 2);
  let k = 0;
  for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) { verts[k++] = warp(i / nx); verts[k++] = warp(j / ny); }
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

// Boat: motor launch with rounded hull (white topsides over a red bottom), sheer line rising to
// the bow, deck with cockpit, cabin with window band and raked windshield, roof and mast.
// Flat shaded. Local frame: x forward, y starboard, z up; z = 0 is the waterline.
const boatVao = G.createVertexArray()!;
let boatVertCount = 0;
function buildBoat() {
  const v: number[] = [];
  type P = [number, number, number];
  const tri = (a: P, b: P, c: P, col: P) => { for (const q of [a, b, c]) v.push(...q, ...col); };
  const quad = (a: P, b: P, c: P, d: P, col: P) => { tri(a, b, c, col); tri(a, c, d, col); };
  const box = (x0: number, x1: number, y0: number, y1: number, z0: number, z1: number, col: P, top: P = col) => {
    quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col);
    quad([x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0], col);
    quad([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], col);
    quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], col);
    quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], top);
  };
  const BOTTOM: P = [0.70, 0.14, 0.10], TOPSIDE: P = [0.96, 0.96, 0.94], STRIPE: P = [0.10, 0.22, 0.40];
  const DECK: P = [0.80, 0.64, 0.42], COCKPIT: P = [0.55, 0.42, 0.28], CABIN: P = [0.97, 0.97, 0.95];
  const GLASS: P = [0.15, 0.22, 0.30], ROOF: P = [0.78, 0.78, 0.74], METAL: P = [0.62, 0.64, 0.68];
  const N = 18;
  const sect = (i: number) => {
    const t = i / N;                                   // 0 = stern, 1 = bow
    const x = -0.5 + t;
    const w = 0.32 * Math.pow(Math.sin(Math.PI * (0.10 + 0.90 * t * 0.93)), 0.65) * (1 - 0.3 * t * t);
    const k = 0.15 * (0.55 + 0.45 * Math.sin(Math.PI * Math.min(t * 1.2, 1))) * (1 - 0.7 * t * t * t);
    const g = 0.15 + 0.10 * t * t;                     // sheer rises toward the bow
    return {
      GL: [x, -w, g] as P, GR: [x, w, g] as P,
      SL: [x, -w * 1.02, 0.03] as P, SR: [x, w * 1.02, 0.03] as P,
      CL: [x, -w * 0.97, -0.02] as P, CR: [x, w * 0.97, -0.02] as P,
      BL: [x, -w * 0.55, -k * 0.75] as P, BR: [x, w * 0.55, -k * 0.75] as P,
      K: [x, 0, -k] as P,
    };
  };
  const secs = Array.from({ length: N + 1 }, (_, i) => sect(i));
  for (let i = 0; i < N; i++) {
    const a = secs[i], b = secs[i + 1];
    quad(a.GL, a.SL, b.SL, b.GL, TOPSIDE); quad(a.GR, b.GR, b.SR, a.SR, TOPSIDE);
    quad(a.SL, a.CL, b.CL, b.SL, STRIPE);  quad(a.SR, b.SR, b.CR, a.CR, STRIPE);
    quad(a.CL, a.BL, b.BL, b.CL, BOTTOM);  quad(a.CR, b.CR, b.BR, a.BR, BOTTOM);
    quad(a.BL, a.K, b.K, b.BL, BOTTOM);    quad(a.BR, b.BR, b.K, a.K, BOTTOM);
    quad(a.GL, b.GL, b.GR, a.GR, DECK);
  }
  const st = secs[0]; // transom
  for (const [p0, p1] of [[st.GL, st.SL], [st.SL, st.CL], [st.CL, st.BL], [st.BL, st.K], [st.K, st.BR], [st.BR, st.CR], [st.CR, st.SR], [st.SR, st.GR]] as [P, P][]) {
    tri([st.GL[0], 0, 0.1], p0, p1, TOPSIDE);
  }
  tri(st.GL, [st.GL[0], 0, 0.1], st.GR, TOPSIDE);
  box(-0.47, -0.18, -0.2, 0.2, 0.08, 0.151, COCKPIT);
  const cx0 = -0.18, cx1 = 0.16, cy = 0.2, g0 = 0.15;
  box(cx0, cx1, -cy, cy, g0, g0 + 0.12, CABIN);
  box(cx0 + 0.01, cx1 - 0.03, -cy + 0.005, cy - 0.005, g0 + 0.12, g0 + 0.22, GLASS);
  quad([cx1 - 0.03, -cy, g0 + 0.12], [cx1 + 0.07, -cy * 0.9, g0 + 0.12], [cx1 - 0.03, -cy * 0.9, g0 + 0.24], [cx1 - 0.03, -cy, g0 + 0.24], GLASS);
  quad([cx1 - 0.03, cy, g0 + 0.12], [cx1 - 0.03, cy, g0 + 0.24], [cx1 - 0.03, cy * 0.9, g0 + 0.24], [cx1 + 0.07, cy * 0.9, g0 + 0.12], GLASS);
  quad([cx1 + 0.07, -cy * 0.9, g0 + 0.12], [cx1 + 0.07, cy * 0.9, g0 + 0.12], [cx1 - 0.03, cy * 0.9, g0 + 0.24], [cx1 - 0.03, -cy * 0.9, g0 + 0.24], GLASS);
  box(cx0 - 0.02, cx1 - 0.01, -cy - 0.02, cy + 0.02, g0 + 0.22, g0 + 0.245, ROOF);
  // Flybridge rail on the roof (the helm camera sits just behind it) and a stern mast.
  box(cx0 + 0.01, cx1 - 0.02, -cy - 0.015, -cy - 0.005, g0 + 0.245, g0 + 0.30, METAL);
  box(cx0 + 0.01, cx1 - 0.02, cy + 0.005, cy + 0.015, g0 + 0.245, g0 + 0.30, METAL);
  box(cx1 - 0.03, cx1 - 0.02, -cy - 0.015, cy + 0.015, g0 + 0.245, g0 + 0.30, METAL);
  box(cx1 - 0.06, cx1 - 0.05, -0.005, 0.005, g0 + 0.245, g0 + 0.40, METAL);
  box(cx1 - 0.065, cx1 - 0.045, -0.01, 0.01, g0 + 0.40, g0 + 0.42, [1.0, 0.3, 0.2]);
  box(0.30, 0.44, -0.012, 0.012, 0.25, 0.27, METAL);

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

// Spray particles: N particles stored in two W x H textures.
const PART_W = 128, PART_H = isMobile ? 32 : 64, PART_N = PART_W * PART_H;
const partVao = G.createVertexArray()!; // no attributes; uses gl_VertexID

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
interface MRT { a: FBO; b: FBO; fb: WebGLFramebuffer; }
function createMRT(w: number, h: number): MRT {
  const a = createFBO(w, h), b = createFBO(w, h);
  const fb = G.createFramebuffer()!;
  G.bindFramebuffer(G.FRAMEBUFFER, fb);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT0, G.TEXTURE_2D, a.tex, 0);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT1, G.TEXTURE_2D, b.tex, 0);
  G.drawBuffers([G.COLOR_ATTACHMENT0, G.COLOR_ATTACHMENT1]);
  if (G.checkFramebufferStatus(G.FRAMEBUFFER) !== G.FRAMEBUFFER_COMPLETE) throw new Error('MRT incomplete');
  return { a, b, fb };
}
let partRead = createMRT(PART_W, PART_H), partWrite = createMRT(PART_W, PART_H);

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
const pBoat = new Program(FRAG_BOAT, VERT_BOAT, ['aPos', 'aCol']);
const pBoatInfo = new Program(BOAT_INFO);
const pPartUpdate = new Program(PARTICLE_UPDATE);
const pBallPush = new Program(BALL_PUSH);
const pBallVel = new Program(BALL_VEL);
const pPart = new Program(FRAG_PARTICLE, VERT_PARTICLE, []);
const pShift = new Program(SHIFT);
const pSky = new Program(SKY_PASS);

// ---------- State ----------
// dye: 2D view's surface shading field. wave: 3D height field (x = h, y = dh/dt, z = splash).
let velocity!: DoubleFBO, pressure!: DoubleFBO, divergence!: FBO, viscB!: FBO, dye!: DoubleFBO, wave!: DoubleFBO;
const boatState = createFBO(1, 1);   // (x, y, heading, speed uv/frame) uploaded from the CPU
const boatInfo = createFBO(2, 1);    // (z, pitch, roll) and water velocity, read back to the CPU
const BOAT_SCALE = 0.34;
const BOAT_DRAFT = 0.15; // keel depth in boat units
let simW = 0, simH = 0, dyeW = 0, dyeH = 0;
let aspect = 1;
// World origin of the sim window (the window scrolls to follow the boat).
const origin = { x: 0, y: 0 };

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (w === 0 || h === 0) return;
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
const swellSlider = $<HTMLInputElement>('swell');
const curlSlider = $<HTMLInputElement>('curl');
const speedSlider = $<HTMLInputElement>('speed');
const speedOut = $<HTMLOutputElement>('speedOut');
const boatBtn = $<HTMLButtonElement>('boat');
const soundBtn = $<HTMLButtonElement>('sound');
$('hideui').addEventListener('click', () => document.body.classList.add('hideui'));
$('showui').addEventListener('click', () => document.body.classList.remove('hideui'));
const dampButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#damp button'));
const viewButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#view button'));
const hint = $<HTMLDivElement>('hint');
const knotsEl = $<HTMLDivElement>('knots');
const hdgEl = $<HTMLDivElement>('hdg');
const fpsEl = $<HTMLDivElement>('fps');

type View = '2d' | '3d' | 'helm';
let view: View = '2d';
const view3D = () => view !== '2d';
let showBoat = true;
const audio = new SeaAudio();
boatBtn.addEventListener('click', () => { showBoat = !showBoat; boatBtn.classList.toggle('active', showBoat); });
soundBtn.addEventListener('click', () => soundBtn.classList.toggle('active', audio.toggle()));
function speedFactor() { return Number(speedSlider.value) / 100; }
function updateSpeedLabel() { speedOut.value = speedFactor().toFixed(1) + 'x'; }
speedSlider.addEventListener('input', updateSpeedLabel);
updateSpeedLabel();

// [velocity dissipation, wave damping]
const DAMP_LEVELS: [number, number][] = [[1.0, 0.998], [0.995, 0.99], [0.975, 0.97], [0.93, 0.92]];
let dampLevel = 1;
let paused = false;

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
const HINTS: Record<View, string> = {
  '2d': 'Drag with the mouse (or touch) to push the water. Walls reflect.',
  '3d': 'Drag to push water. Two fingers / right-drag to orbit, pinch or wheel to zoom.',
  'helm': 'W/S or ↑/↓ throttle, A/D or ←/→ steer. Touch: left half steers, right half is the throttle.',
};
function setView(v: View) {
  view = v;
  viewButtons.forEach((x) => x.classList.toggle('active', x.dataset.view === v));
  document.body.classList.toggle('view3d', view3D());
  document.body.classList.toggle('helm', v === 'helm');
  hint.textContent = HINTS[v];
  pointers.clear();
  input.thr = 0; input.str = 0;
  if (v === 'helm' && showBoat === false) { showBoat = true; boatBtn.classList.add('active'); }
}
viewButtons.forEach((b) => b.addEventListener('click', () => setView(b.dataset.view as View)));
function clearAll() {
  for (const f of [velocity.read, velocity.write, pressure.read, pressure.write, divergence, viscB, dye.read, dye.write,
    wave.read, wave.write, partRead.a, partRead.b, partWrite.a, partWrite.b]) {
    G.bindFramebuffer(G.FRAMEBUFFER, f.fb); G.clearColor(0, 0, 0, 0); G.clear(G.COLOR_BUFFER_BIT);
  }
  boat.x = 0.5; boat.y = 0.5; boat.vx = 0; boat.vy = 0; boat.yawRate = 0;
}
$('reset').addEventListener('click', clearAll);
const pauseBtn = $<HTMLButtonElement>('pause');
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; });

function currentHeightScale() { return 0.06 + (Number(heightSlider.value) / 100) * 0.5; }
function swellAmp() { return Number(swellSlider.value) / 100 * 0.5; } // height units at full slider
function chop() { return Number(curlSlider.value) / 100 * 2.6; }
let simTime = 0;
function swellUniforms(p: Program) {
  return p.f('uSwell', view3D() ? swellAmp() : 0).f('uSimTime', simTime).f('uChop', chop()).v2('uOrigin', origin.x, origin.y);
}

// ---------- Boat (CPU physics) ----------
const boat = {
  x: 0.5, y: 0.5,            // position in window uv
  heading: Math.PI / 2,      // radians, 0 = +x (east), pi/2 = +y (north / up the screen)
  vx: 0, vy: 0,              // velocity, world units per frame
  yawRate: 0, bank: 0,
  z: 0, pitch: 0, roll: 0,   // from the GPU readback (world z of the hull centre, radians)
  cx: 0, cy: 0,              // displaced hull centre xy (world, from the GPU readback)
  zHist: [0, 0, 0],
  curX: 0, curY: 0,          // water velocity under the hull (sim texels / frame)
  slam: 0,                   // hull impact this frame, 0..1
};
const ctrl = { throttle: 0, steer: 0 };   // smoothed
const input = { thr: 0, str: 0 };         // raw target from keys / touch
const keys = new Set<string>();
const TOP_SPEED = 0.012;                   // world units / frame at full throttle

const boatBuf = HALF ? new Uint16Array(4) : new Float32Array(4);
function uploadBoat() {
  const sp = Math.hypot(boat.vx, boat.vy) * 0.5; // uv-ish units per frame for the wake shader
  const vals = [boat.x, boat.y, boat.heading, sp];
  if (HALF) for (let i = 0; i < 4; i++) (boatBuf as Uint16Array)[i] = toHalf(vals[i]);
  else for (let i = 0; i < 4; i++) (boatBuf as Float32Array)[i] = vals[i];
  G.bindTexture(G.TEXTURE_2D, boatState.tex);
  G.texSubImage2D(G.TEXTURE_2D, 0, 0, 0, 1, 1, G.RGBA, FMT.type, boatBuf);
}
const infoBuf = HALF ? new Uint16Array(8) : new Float32Array(8);
function readBoatInfo() {
  const hs = currentHeightScale();
  pBoatInfo.use().f('uAspect', aspect).f('uHeightScale', hs).f('uScale', BOAT_SCALE)
    .tex('uHeight', wave.read.attach(0)).tex('uVelocity', velocity.read.attach(1)).tex('uBoat', boatState.attach(2));
  swellUniforms(pBoatInfo);
  blit(boatInfo);
  G.readPixels(0, 0, 2, 1, G.RGBA, FMT.type, infoBuf);
  const rd = (i: number) => HALF ? fromHalf((infoBuf as Uint16Array)[i]) : (infoBuf as Float32Array)[i];
  const z = rd(0), pitch = rd(1), roll = rd(2), cx = rd(4), cy = rd(5), wx = rd(6), wy = rd(7);
  if ([z, pitch, roll, cx, cy, wx, wy].every(Number.isFinite)) {
    boat.z = z; boat.pitch = pitch; boat.roll = roll; boat.curX = cx; boat.curY = cy; boat.cx = wx; boat.cy = wy;
  }
}

function wrapAngle(a: number) { return Math.atan2(Math.sin(a), Math.cos(a)); }

function boatPhysics(dt: number) {
  const driving = view === 'helm';
  // Smoothed controls: throttle spools up, steering is quick.
  const tgtThr = driving ? input.thr : 0, tgtStr = driving ? input.str : 0;
  ctrl.throttle += (tgtThr - ctrl.throttle) * (1 - Math.exp(-dt / 10));
  ctrl.steer += (tgtStr - ctrl.steer) * (1 - Math.exp(-dt / 4));

  const fx = Math.cos(boat.heading), fy = Math.sin(boat.heading);
  const rx = fy, ry = -fx; // starboard
  const fwdSpeed = boat.vx * fx + boat.vy * fy;
  // Propulsion (reverse is weaker), plus surfing down wave faces.
  const accel = ctrl.throttle * (ctrl.throttle > 0 ? 0.00042 : 0.00022);
  const surf = -Math.sin(boat.pitch) * 0.0005;
  boat.vx += fx * (accel + surf) * dt; boat.vy += fy * (accel + surf) * dt;
  // Water current carries the boat.
  boat.vx += boat.curX * (2 * aspect / simW) * 0.02 * dt; boat.vy += boat.curY * (2 / simH) * 0.02 * dt;
  // Drag (linear + quadratic) and lateral grip so the hull slides a little in turns.
  const sp = Math.hypot(boat.vx, boat.vy);
  const drag = Math.min(0.95, (0.012 + sp * 1.6) * dt);
  boat.vx *= 1 - drag; boat.vy *= 1 - drag;
  const lat = boat.vx * rx + boat.vy * ry;
  const grip = 1 - Math.exp(-dt / 5);
  boat.vx -= rx * lat * grip; boat.vy -= ry * lat * grip;
  // Steering: yaw rate builds with speed, with a little inertia. Reversing flips the helm.
  const steerAuthority = Math.max(-1, Math.min(1, fwdSpeed / (TOP_SPEED * 0.35)));
  const tgtYaw = ctrl.steer * 0.045 * steerAuthority;
  boat.yawRate += (tgtYaw - boat.yawRate) * (1 - Math.exp(-dt / 6));
  boat.heading = wrapAngle(boat.heading + boat.yawRate * dt);
  // Bank into turns.
  const tgtBank = -ctrl.steer * Math.min(Math.abs(fwdSpeed) / TOP_SPEED, 1) * 0.22;
  boat.bank += (tgtBank - boat.bank) * (1 - Math.exp(-dt / 8));
  // Integrate position (window uv).
  boat.x += boat.vx * dt / (2 * aspect);
  boat.y += boat.vy * dt / 2;
  if (!driving) { // keep an idle boat inside the window without scrolling the world
    boat.x = Math.min(0.94, Math.max(0.06, boat.x)); boat.y = Math.min(0.94, Math.max(0.06, boat.y));
  }
  // Hull slam: vertical acceleration of the hull centre.
  boat.zHist.push(boat.z); boat.zHist.shift();
  const acc = boat.zHist[2] - 2 * boat.zHist[1] + boat.zHist[0];
  const slamNow = Math.min(Math.abs(acc) * 60, 1);
  boat.slam = Math.max(boat.slam * Math.pow(0.85, dt), slamNow);
}

// Scroll the simulation window so the boat stays near its centre (endless ocean).
let pendingShiftW = { x: 0, y: 0 };
function recenterIfNeeded() {
  if (Math.abs(boat.x - 0.5) < 0.22 && Math.abs(boat.y - 0.5) < 0.22) return;
  const sx = 0.5 - boat.x, sy = 0.5 - boat.y;
  const shiftW = { x: sx * 2 * aspect, y: sy * 2 };
  for (const f of [velocity, pressure, wave, dye]) {
    pShift.use().v2('uShift', sx, sy).tex('uSrc', f.read.attach(0));
    blit(f.write); f.swap();
  }
  origin.x -= shiftW.x; origin.y -= shiftW.y;
  boat.x = 0.5; boat.y = 0.5;
  boat.cx += shiftW.x; boat.cy += shiftW.y;
  pendingShiftW = shiftW; // particles are moved in the next update pass
  for (const p of pointers.values()) { p.x += sx; p.y += sy; }
}

// ---------- Camera ----------
interface Camera { eye: [number, number, number]; fwd: [number, number, number]; right: [number, number, number]; up: [number, number, number]; fov: number; }
const orbitCam = { yaw: 0.0, pitch: 0.95, dist: 3.2 };
const helmCam = { yaw: Math.PI / 2, shake: 0, sx: 0, sy: 0, sz: 0 };
const FOV = 45 * Math.PI / 180;
function norm(v: [number, number, number]): [number, number, number] { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function orbitCamera(): Camera {
  const cp = Math.cos(orbitCam.pitch);
  const eye: [number, number, number] = [orbitCam.dist * cp * Math.sin(orbitCam.yaw), -orbitCam.dist * cp * Math.cos(orbitCam.yaw), orbitCam.dist * Math.sin(orbitCam.pitch)];
  const fwd = norm([-eye[0], -eye[1], -eye[2]]);
  const right = norm(cross(fwd, [0, 0, 1]));
  const up = cross(right, fwd);
  return { eye, fwd, right, up, fov: FOV };
}
// Boat frame from heading / pitch / roll / bank.
function boatBasis() {
  const h = boat.heading, p = boat.pitch, r = boat.roll + boat.bank;
  const f: [number, number, number] = [Math.cos(h), Math.sin(h), 0];
  const rt: [number, number, number] = [Math.sin(h), -Math.cos(h), 0];
  const u: [number, number, number] = [0, 0, 1];
  const fp: [number, number, number] = [f[0] * Math.cos(p) + u[0] * Math.sin(p), f[1] * Math.cos(p) + u[1] * Math.sin(p), f[2] * Math.cos(p) + u[2] * Math.sin(p)];
  const up1: [number, number, number] = [u[0] * Math.cos(p) - f[0] * Math.sin(p), u[1] * Math.cos(p) - f[1] * Math.sin(p), u[2] * Math.cos(p) - f[2] * Math.sin(p)];
  const rr: [number, number, number] = [rt[0] * Math.cos(r) + up1[0] * Math.sin(r), rt[1] * Math.cos(r) + up1[1] * Math.sin(r), rt[2] * Math.cos(r) + up1[2] * Math.sin(r)];
  const uu = norm(cross(rr, fp));
  return { f: fp, r: rr, u: uu };
}
function boatWorld(): [number, number, number] {
  return [boat.cx, boat.cy, boat.z];
}
function helmCamera(dt: number): Camera {
  // The camera heading lags the hull a little so turns feel weighty.
  helmCam.yaw = wrapAngle(helmCam.yaw + wrapAngle(boat.heading - helmCam.yaw) * (1 - Math.exp(-dt / 7)));
  const { f, r, u } = boatBasis();
  const c = boatWorld();
  const S = BOAT_SCALE;
  const off = { x: qn('camx', -0.72) * S, z: qn('camz', 0.78) * S };
  // Camera shake from hull slams.
  helmCam.shake = Math.max(helmCam.shake * Math.pow(0.8, dt), boat.slam * 0.012);
  const sh = helmCam.shake;
  helmCam.sx += ((Math.random() - 0.5) * sh - helmCam.sx) * 0.5;
  helmCam.sy += ((Math.random() - 0.5) * sh - helmCam.sy) * 0.5;
  helmCam.sz += ((Math.random() - 0.5) * sh - helmCam.sz) * 0.5;
  const eye: [number, number, number] = [
    c[0] + f[0] * off.x + u[0] * off.z + helmCam.sx,
    c[1] + f[1] * off.x + u[1] * off.z + helmCam.sy,
    c[2] + f[2] * off.x + u[2] * off.z + helmCam.sz,
  ];
  // Look direction: lagged yaw, softened pitch, rolled with the hull and the bank.
  const cp = qn('camp', -0.40) + boat.pitch * 0.45;
  const cy = helmCam.yaw;
  let fwd: [number, number, number] = norm([Math.cos(cy) * Math.cos(cp), Math.sin(cy) * Math.cos(cp), Math.sin(cp)]);
  const roll = (boat.roll + boat.bank * 1.0) * 0.75;
  let right = norm(cross(fwd, [0, 0, 1]));
  let up = cross(right, fwd);
  const rr = norm([right[0] * Math.cos(roll) + up[0] * Math.sin(roll), right[1] * Math.cos(roll) + up[1] * Math.sin(roll), right[2] * Math.cos(roll) + up[2] * Math.sin(roll)]);
  right = rr; up = norm(cross(right, fwd));
  void r;
  const speed = Math.hypot(boat.vx, boat.vy) / TOP_SPEED;
  const fov = (58 + 18 * Math.min(speed, 1.2)) * Math.PI / 180;
  return { eye, fwd, right, up, fov };
}
function camMatrices(cam: Camera): Mat4 {
  const tgt: [number, number, number] = [cam.eye[0] + cam.fwd[0], cam.eye[1] + cam.fwd[1], cam.eye[2] + cam.fwd[2]];
  return multiply(perspective(cam.fov, aspect, 0.02, 60), lookAt(cam.eye, tgt, cam.up));
}
// Screen (0..1, y down) -> sim uv by intersecting the orbit camera ray with the z = 0 plane.
function unproject(sx: number, sy: number): [number, number] | null {
  const cam = orbitCamera();
  const th = Math.tan(cam.fov / 2);
  const nx = (sx * 2 - 1) * th * aspect, ny = (1 - sy * 2) * th;
  const d = [cam.fwd[0] + nx * cam.right[0] + ny * cam.up[0], cam.fwd[1] + nx * cam.right[1] + ny * cam.up[1], cam.fwd[2] + nx * cam.right[2] + ny * cam.up[2]];
  if (d[2] >= -1e-5) return null;
  const t = -cam.eye[2] / d[2];
  const px = cam.eye[0] + d[0] * t, py = cam.eye[1] + d[1] * t;
  return [px / (2 * aspect) + 0.5, py / 2 + 0.5];
}

// ---------- Pointer / keyboard input ----------
interface Pointer { x: number; y: number; dx: number; dy: number; moved: boolean; sx: number; sy: number; orbit: boolean; age: number; zone: 'steer' | 'throttle' | null; sx0: number; sy0: number; }
const pointers = new Map<number, Pointer>();
let pinchDist = 0;

function screenPos(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}
function toSim(sx: number, sy: number): [number, number] | null {
  return view3D() ? unproject(sx, sy) : [sx, 1 - sy];
}
function updateDriveInput() {
  let str = 0, thr = 0, hasS = false, hasT = false;
  for (const p of pointers.values()) {
    if (p.zone === 'steer') { str = Math.max(-1, Math.min(1, -(p.sx - p.sx0) / 0.18)); hasS = true; }
    if (p.zone === 'throttle') { thr = Math.max(-0.5, Math.min(1, -(p.sy - p.sy0) / 0.22)); hasT = true; }
  }
  const kStr = (keys.has('a') || keys.has('ArrowLeft') ? 1 : 0) - (keys.has('d') || keys.has('ArrowRight') ? 1 : 0);
  const kThr = (keys.has('w') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('s') || keys.has('ArrowDown') ? 0.5 : 0);
  input.str = hasS ? str : kStr;
  input.thr = hasT ? thr : kThr;
}
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('pointerdown', (e) => {
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic events have no active pointer */ }
  const [sx, sy] = screenPos(e);
  const p = toSim(sx, sy) ?? [0, 0];
  const orbit = view === '3d' && (e.button === 2 || e.shiftKey);
  const zone = view === 'helm' ? (sx < 0.5 ? 'steer' : 'throttle') : null;
  pointers.set(e.pointerId, { x: p[0], y: p[1], dx: 0, dy: 0, moved: false, sx, sy, orbit, age: 0, zone, sx0: sx, sy0: sy });
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinchDist = Math.hypot(a.sx - b.sx, a.sy - b.sy);
  }
  updateDriveInput();
});
canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const [sx, sy] = screenPos(e);
  if (p.zone) { p.sx = sx; p.sy = sy; updateDriveInput(); return; }
  const multiTouch = view === '3d' && pointers.size >= 2;
  if (p.orbit || multiTouch) {
    const scale = multiTouch ? 0.5 : 1;
    orbitCam.yaw += (sx - p.sx) * 3.0 * scale;
    orbitCam.pitch = Math.min(1.5, Math.max(0.15, orbitCam.pitch + (sy - p.sy) * 2.0 * scale));
    p.sx = sx; p.sy = sy;
    if (multiTouch) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.sx - b.sx, a.sy - b.sy);
      if (pinchDist > 0) orbitCam.dist = Math.min(8, Math.max(1.2, orbitCam.dist * (pinchDist / d)));
      pinchDist = d;
    }
    return;
  }
  const s = toSim(sx, sy);
  p.sx = sx; p.sy = sy;
  if (!s) return;
  p.dx += s[0] - p.x; p.dy += s[1] - p.y; p.x = s[0]; p.y = s[1]; p.moved = true;
});
const end = (e: PointerEvent) => { pointers.delete(e.pointerId); pinchDist = 0; updateDriveInput(); };
canvas.addEventListener('pointerup', end);
canvas.addEventListener('pointercancel', end);
canvas.addEventListener('pointerleave', end);
canvas.addEventListener('wheel', (e) => {
  if (view !== '3d') return;
  e.preventDefault();
  orbitCam.dist = Math.min(8, Math.max(1.2, orbitCam.dist * Math.exp(e.deltaY * 0.001)));
}, { passive: false });
window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) { keys.add(k); e.preventDefault(); updateDriveInput(); }
  if (k === 'h' || k === 'H') setView(view === 'helm' ? '3d' : 'helm');
});
window.addEventListener('keyup', (e) => {
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keys.delete(k); updateDriveInput();
});
window.addEventListener('blur', () => { keys.clear(); updateDriveInput(); });

// ---------- Simulation step ----------
function ballRadiusUv() { return 0.04 + (Number(splatSlider.value) / 100) * 0.14; }
const BALL_SINK_FRAC = 0.55;

// Apply the 3D ball push for one pointer this frame (pressing and/or moving).
function ballPush(p: Pointer, dt: number) {
  const R = ballRadiusUv();
  const Rw = R * 2;
  const hs = currentHeightScale();
  const depth = (Rw * (1 - BALL_SINK_FRAC)) / hs;
  const entering = Math.exp(-p.age / 5);
  const speed = Math.hypot(p.dx, p.dy);
  let ax = p.dx * aspect, ay = p.dy;
  const m = Math.hypot(ax, ay);
  if (m > 1e-6) { ax /= m; ay /= m; } else { ax = 0; ay = 0; }
  const ridge = depth * 0.25 * entering * dt;
  const move = Math.min(speed * 8, 0.12) * depth;
  pBallPush.use().v2('uTexel', 1 / dyeW, 1 / dyeH).v2('uPoint', p.x, p.y).f('uRadius', R).f('uDepth', depth)
    .f('uRidge', ridge).v2('uDir', ax, ay).f('uMove', move).f('uAspect', aspect).tex('uTarget', wave.read.attach(0));
  blit(wave.write); wave.swap();
  const press = 10 * entering * dt;
  const mf = Math.min(speed, 0.04) * 300;
  pBallVel.use().v2('uTexel', 1 / simW, 1 / simH).v2('uPoint', p.x, p.y).f('uRadius', R).f('uPress', press)
    .v2('uMoveForce', ax / aspect * mf, ay * mf).f('uAspect', aspect).tex('uTarget', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();
}

function splat(x: number, y: number, dx: number, dy: number) {
  const radius = (Number(splatSlider.value) / 100) * 0.01 + 0.0004;
  const force = 1400;
  const speed = Math.hypot(dx, dy);
  pSplat.use().v2('uTexel', 1 / simW, 1 / simH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
    .tex('uTarget', velocity.read.attach(0)).v3('uValue', dx * force, dy * force, 0);
  blit(velocity.write); velocity.swap();
  pSplat.use().v2('uTexel', 1 / dyeW, 1 / dyeH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
    .tex('uTarget', dye.read.attach(0)).v3('uValue', Math.min(speed * 40, 1.2), 0, 0);
  blit(dye.write); dye.swap();
}

function step(dt0: number) {
  const tx = 1 / simW, ty = 1 / simH;
  const sf = speedFactor();
  const dt = dt0 * sf; // fluid time step
  const [velDamp, waveDamp] = DAMP_LEVELS[dampLevel];

  for (const p of pointers.values()) {
    if (p.zone || p.orbit) continue;
    if (view === '3d') {
      const mm = Math.hypot(p.dx, p.dy), cap = 0.04;
      if (mm > cap) { p.dx *= cap / mm; p.dy *= cap / mm; }
      ballPush(p, dt0);
      p.age += dt0;
      p.dx = 0; p.dy = 0; p.moved = false;
      continue;
    }
    if (p.moved && (p.dx !== 0 || p.dy !== 0)) {
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

  // 2. Viscous diffusion.
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

  // 3-7. Boundary, divergence, pressure solve, projection.
  pBound.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();
  pDiv.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(divergence);
  pJacobi.use().v2('uTexel', tx, ty).f('uAlpha', -1).f('uRBeta', 0.25).tex('uB', divergence.attach(1));
  for (let i = 0; i < PRESSURE_ITERS; i++) {
    pJacobi.tex('uX', pressure.read.attach(0));
    blit(pressure.write); pressure.swap();
  }
  pGrad.use().v2('uTexel', tx, ty).tex('uPressure', pressure.read.attach(0)).tex('uVelocity', velocity.read.attach(1));
  blit(velocity.write); velocity.swap();
  pBound.use().v2('uTexel', tx, ty).tex('uVelocity', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();

  // 8. Boat: physics on the CPU, state to the GPU, attitude back from the GPU.
  boatPhysics(dt0);
  if (view === 'helm') recenterIfNeeded();
  uploadBoat();

  if (!view3D()) {
    pAdvect.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', dt * (dyeW / simW)).f('uDissipation', 0.985)
      .tex('uVelocity', velocity.read.attach(0)).tex('uSource', dye.read.attach(1));
    blit(dye.write); dye.swap();
    readBoatInfo();
    return;
  }

  // 9 (3D). Advect the height field with the flow.
  pAdvect.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', dt * (dyeW / simW)).f('uDissipation', 1.0)
    .tex('uVelocity', velocity.read.attach(0)).tex('uSource', wave.read.attach(1));
  blit(wave.write); wave.swap();

  // 10 (3D). Surface waves, sub-stepped.
  const subs = Math.max(1, Math.round(2 * sf));
  const wdt = dt0;
  const c2 = Math.min(0.3, 0.4 / Math.max(wdt * wdt, 1e-3));
  const wd = Math.pow(waveDamp * (1 - 0.01 * nu), 1 / subs);
  const heightScale = currentHeightScale();
  pWave.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', wdt).f('uC2', c2).f('uDamp', wd)
    .f('uAspect', aspect).f('uBoatSize', showBoat ? BOAT_SCALE * 0.25 : 0.0001)
    .f('uDraft', BOAT_SCALE * BOAT_DRAFT / heightScale)
    .tex('uBoat', boatState.attach(1));
  swellUniforms(pWave);
  for (let i = 0; i < subs; i++) {
    pWave.tex('uField', wave.read.attach(0));
    blit(wave.write); wave.swap();
  }

  // 11 (3D). Spray particles.
  pPartUpdate.use().f('uDt', dt0).f('uTime', (performance.now() % 100000) / 1000).f('uAspect', aspect)
    .f('uHeightScale', heightScale).f('uBoatSize', BOAT_SCALE * 0.25).f('uSpawnRate', showBoat ? 1 : 0)
    .v2('uSimTexel', tx, ty).v2('uShiftW', pendingShiftW.x, pendingShiftW.y)
    .tex('uA', partRead.a.attach(0)).tex('uB', partRead.b.attach(1))
    .tex('uField', wave.read.attach(2)).tex('uBoat', boatState.attach(3))
    .tex('uVelocity', velocity.read.attach(4));
  swellUniforms(pPartUpdate);
  pendingShiftW = { x: 0, y: 0 };
  G.bindVertexArray(quadVao);
  G.bindFramebuffer(G.FRAMEBUFFER, partWrite.fb);
  G.viewport(0, 0, PART_W, PART_H);
  G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  const t = partRead; partRead = partWrite; partWrite = t;

  readBoatInfo();
}

// ---------- Rendering ----------
function drawBoat(vp: Mat4, eye: [number, number, number], heightScale: number) {
  if (!showBoat) return;
  pBoat.use().m4('uVP', vp).v2('uTexel', 1 / dyeW, 1 / dyeH).f('uAspect', aspect).f('uHeightScale', heightScale)
    .f('uScale', BOAT_SCALE).v3('uEye', eye[0], eye[1], eye[2]).f('uBank', boat.bank)
    .tex('uState', boatState.attach(0)).tex('uHeight', (view3D() ? wave : dye).read.attach(1));
  swellUniforms(pBoat);
  G.bindVertexArray(boatVao);
  G.drawArrays(G.TRIANGLES, 0, boatVertCount);
  G.bindVertexArray(quadVao);
}

function render2D(t: number) {
  pDisplay.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uTime', t)
    .tex('uDye', dye.read.attach(0)).tex('uPressure', pressure.read.attach(1)).tex('uVelocity', velocity.read.attach(2));
  blit(null);
  G.enable(G.DEPTH_TEST);
  G.clear(G.DEPTH_BUFFER_BIT);
  drawBoat(ortho(-aspect, aspect, -1, 1, -10, 10), [0, 0, 10], 0.0);
  G.disable(G.DEPTH_TEST);
}

function drawSpray(vp: Mat4) {
  pPart.use().m4('uVP', vp).f('uPointScale', canvas.height * 0.012).tex('uA', partRead.a.attach(0)).i('uTexW', PART_W);
  G.enable(G.BLEND);
  G.blendFunc(G.SRC_ALPHA, G.ONE_MINUS_SRC_ALPHA);
  G.depthMask(false);
  G.bindVertexArray(partVao);
  G.drawArrays(G.POINTS, 0, PART_N);
  G.bindVertexArray(quadVao);
  G.depthMask(true);
  G.disable(G.BLEND);
}

function render3D(dt: number) {
  const cam = view === 'helm' ? helmCamera(dt) : orbitCamera();
  const vp = camMatrices(cam);
  const heightScale = currentHeightScale();
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.viewport(0, 0, canvas.width, canvas.height);
  G.disable(G.DEPTH_TEST);
  // Sky.
  pSky.use().v3('uFwd', ...cam.fwd).v3('uRight', ...cam.right).v3('uUp', ...cam.up)
    .f('uTanHalfFov', Math.tan(cam.fov / 2)).f('uAspect2', aspect);
  blit(null);
  G.enable(G.DEPTH_TEST);
  G.clear(G.DEPTH_BUFFER_BIT);
  // Ocean.
  pMesh.use().m4('uVP', vp).f('uAspect', aspect).f('uHeightScale', heightScale)
    .v2('uTexel', 1 / dyeW, 1 / dyeH).v3('uEye', ...cam.eye).v2('uBoatUv', showBoat ? boat.x : -10, boat.y)
    .tex('uHeight', wave.read.attach(0)).tex('uVelocity', velocity.read.attach(1));
  swellUniforms(pMesh);
  G.bindVertexArray(meshVao);
  G.drawElements(G.TRIANGLES, meshIndexCount, G.UNSIGNED_INT, 0);
  G.bindVertexArray(quadVao);
  drawBoat(vp, cam.eye, heightScale);
  drawSpray(vp);
  G.disable(G.DEPTH_TEST);
}

function updateHud() {
  if (view !== 'helm') return;
  const sp = Math.hypot(boat.vx, boat.vy);
  knotsEl.textContent = (sp * 2700).toFixed(0);
  const deg = ((90 - boat.heading * 180 / Math.PI) % 360 + 360) % 360;
  hdgEl.textContent = `HDG ${deg.toFixed(0).padStart(3, '0')}°`;
}

let fpsN = 0, fpsT = performance.now();
let last = performance.now();
function frame(now: number) {
  fpsN++;
  if (now - fpsT > 500) { fpsEl.textContent = (fpsN * 1000 / (now - fpsT)).toFixed(0) + ' fps'; fpsN = 0; fpsT = now; }
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  resize();
  if (!velocity) { requestAnimationFrame(frame); return; }
  if (!paused) { simTime += dt * speedFactor(); step(dt * 60); }
  if (view3D()) render3D(dt * 60); else render2D(now / 1000);
  updateHud();
  audio.update(ctrl.throttle, Math.min(Math.hypot(boat.vx, boat.vy) / TOP_SPEED, 1), boat.slam, view === 'helm');
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Dev hooks: advance N frames synchronously (works while rAF is throttled) and sample state.
(window as unknown as { __advance: (n: number) => void }).__advance = (n: number) => {
  for (let i = 0; i < n; i++) { simTime += speedFactor() / 60; step(1); if (view === 'helm') helmCamera(1); }
  if (view3D()) render3D(1); else render2D(0);
};
(window as unknown as { __dbg: () => unknown }).__dbg = () => ({ boat: { ...boat }, ctrl: { ...ctrl }, input: { ...input }, origin: { ...origin }, view });
(window as unknown as { __cam: () => unknown }).__cam = () => { const c = view === 'helm' ? helmCamera(0) : orbitCamera(); return { ...c, vp: Array.from(camMatrices(c)).map((x) => +x.toFixed(3)) }; };
