import {
  VERT, ADVECT, SPLAT, DIVERGENCE, JACOBI, GRADIENT_SUBTRACT, BOUNDARY, DISPLAY,
} from './shaders';

// ---------- WebGL setup ----------
const canvas = document.getElementById('c') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance' });
if (!gl) throw new Error('WebGL2 is required');
const G: WebGL2RenderingContext = gl;

// Pick a renderable float texture format. RGBA16F is the most portable; some
// desktops only expose EXT_color_buffer_float (32F), some phones only half-float.
G.getExtension('EXT_color_buffer_float');
G.getExtension('EXT_color_buffer_half_float');
const linearHalf = !!G.getExtension('OES_texture_half_float_linear') || true; // core in WebGL2
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
  probeFormat(G.RGBA16F, G.HALF_FLOAT, linearHalf ? G.LINEAR : G.NEAREST) ??
  probeFormat(G.RGBA32F, G.FLOAT, linearFloat ? G.LINEAR : G.NEAREST);
if (!probed) throw new Error('Float render targets are not supported on this device');
const FMT: TexFormat = probed;

const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || navigator.maxTouchPoints > 1;
const SIM_RES = isMobile ? 160 : 256;
const DYE_RES = isMobile ? 384 : 768;
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
  constructor(frag: string) {
    const p = G.createProgram()!;
    G.attachShader(p, compile(G.VERTEX_SHADER, VERT));
    G.attachShader(p, compile(G.FRAGMENT_SHADER, frag));
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
  tex(name: string, unit: number) { G.uniform1i(this.uniforms.get(name)!, unit); return this; }
}

// Fullscreen triangle-pair
const vao = G.createVertexArray()!;
G.bindVertexArray(vao);
const vbo = G.createBuffer()!;
G.bindBuffer(G.ARRAY_BUFFER, vbo);
G.bufferData(G.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), G.STATIC_DRAW);
G.enableVertexAttribArray(0);
G.vertexAttribPointer(0, 2, G.FLOAT, false, 0, 0);

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
const pDisplay = new Program(DISPLAY);

// ---------- State ----------
let velocity!: DoubleFBO, pressure!: DoubleFBO, divergence!: FBO, viscB!: FBO, dye!: DoubleFBO;
let simW = 0, simH = 0, dyeW = 0, dyeH = 0;

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 1.5 : 2);
  const w = Math.floor(canvas.clientWidth * dpr), h = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width === w && canvas.height === h && velocity) return;
  canvas.width = w; canvas.height = h;
  const aspect = w / h;
  const dims = (res: number) => aspect >= 1
    ? [Math.round(res * aspect), res] : [res, Math.round(res / aspect)];
  [simW, simH] = dims(SIM_RES);
  [dyeW, dyeH] = dims(DYE_RES);
  velocity = createDoubleFBO(simW, simH);
  pressure = createDoubleFBO(simW, simH);
  divergence = createFBO(simW, simH);
  viscB = createFBO(simW, simH);
  dye = createDoubleFBO(dyeW, dyeH);
}
window.addEventListener('resize', resize);
resize();

// ---------- UI ----------
const viscSlider = document.getElementById('visc') as HTMLInputElement;
const viscOut = document.getElementById('viscOut') as HTMLOutputElement;
const splatSlider = document.getElementById('splat') as HTMLInputElement;
const dampButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#damp button'));
const DAMP_LEVELS = [1.0, 0.995, 0.975, 0.93];
let dampLevel = 1;
let paused = false;

function viscosity(): number {
  // 0..100 -> 0..3 (grid units), quadratic so the low end is fine-grained.
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
document.getElementById('reset')!.addEventListener('click', () => {
  for (const f of [velocity.read, velocity.write, pressure.read, pressure.write, divergence, viscB, dye.read, dye.write]) {
    G.bindFramebuffer(G.FRAMEBUFFER, f.fb); G.clear(G.COLOR_BUFFER_BIT);
  }
});
const pauseBtn = document.getElementById('pause') as HTMLButtonElement;
pauseBtn.addEventListener('click', () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; });

// ---------- Pointer input (mouse + touch via Pointer Events) ----------
interface Pointer { x: number; y: number; dx: number; dy: number; moved: boolean; }
const pointers = new Map<number, Pointer>();

function toSim(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height];
}
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  const [x, y] = toSim(e);
  pointers.set(e.pointerId, { x, y, dx: 0, dy: 0, moved: false });
});
canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) return;
  const [x, y] = toSim(e);
  p.dx += x - p.x; p.dy += y - p.y; p.x = x; p.y = y; p.moved = true;
});
const end = (e: PointerEvent) => pointers.delete(e.pointerId);
canvas.addEventListener('pointerup', end);
canvas.addEventListener('pointercancel', end);
canvas.addEventListener('pointerleave', end);

// ---------- Simulation step ----------
function splat(x: number, y: number, dx: number, dy: number) {
  const aspect = canvas.width / canvas.height;
  const radius = (Number(splatSlider.value) / 100) * 0.01 + 0.0004;
  const force = 1800;

  pSplat.use().v2('uTexel', 1 / simW, 1 / simH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
    .tex('uTarget', velocity.read.attach(0)).v3('uValue', dx * force, dy * force, 0);
  blit(velocity.write); velocity.swap();

  const speed = Math.hypot(dx, dy);
  pSplat.use().v2('uTexel', 1 / dyeW, 1 / dyeH).v2('uPoint', x, y).f('uRadius', radius).f('uAspect', aspect)
    .tex('uTarget', dye.read.attach(0)).v3('uValue', Math.min(speed * 40, 1.2), 0, 0);
  blit(dye.write); dye.swap();
}

function step(dt: number) {
  const tx = 1 / simW, ty = 1 / simH;

  // Pointer forcing.
  for (const p of pointers.values()) {
    if (p.moved && (p.dx !== 0 || p.dy !== 0)) splat(p.x, p.y, p.dx, p.dy);
    p.dx = 0; p.dy = 0; p.moved = false;
  }

  // 1. Advect velocity (self-advection) with damping.
  pAdvect.use().v2('uTexel', tx, ty).f('uDt', dt).f('uDissipation', DAMP_LEVELS[dampLevel])
    .tex('uVelocity', velocity.read.attach(0)).tex('uSource', velocity.read.attach(0));
  blit(velocity.write); velocity.swap();

  // 2. Viscous diffusion: (I - nu*dt*Lap) v = v0, solved with Jacobi.
  const nu = viscosity();
  if (nu > 0) {
    const alpha = 1 / (nu * dt);
    const rBeta = 1 / (4 + alpha);
    // Copy the current velocity into viscB (advect with dt = 0 is a plain copy).
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

  // 5. Pressure Poisson: Lap p = div.  alpha = -1, rBeta = 1/4.
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

  // 8. Advect the surface/dye field.
  pAdvect.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uDt', dt * (dyeW / simW)).f('uDissipation', 0.985)
    .tex('uVelocity', velocity.read.attach(0)).tex('uSource', dye.read.attach(1));
  blit(dye.write); dye.swap();
}

function render(t: number) {
  pDisplay.use().v2('uTexel', 1 / dyeW, 1 / dyeH).f('uTime', t)
    .tex('uDye', dye.read.attach(0)).tex('uPressure', pressure.read.attach(1)).tex('uVelocity', velocity.read.attach(2));
  blit(null);
}

let last = performance.now();
function frame(now: number) {
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  resize();
  if (!paused) step(dt * 60); // dt in "frames" so velocity is in texels/frame
  render(now / 1000);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
