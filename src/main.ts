import * as S from './shaders';
import { Mat4, perspective, lookAt, multiply } from './mat4';
import { IC, EYE, SUN_DIR, WIND_DIR, clamp, mix, mix3, smoothstep, stormAt, norm3 } from './world';
import { uWA, uWB, updatePhases, sampleWater, newSample, sigWaveHeight } from './waves';
import { buildLand, buildEyeIsland, nearEye, buildShoreMap, buildBuoy, terrainHeight, PILLARS, ARCH, LIGHTHOUSE, BUOYS, SHORE, BUOY_COLS } from './island';
import {
  Ship, shipAxes, buildFireLauncherMesh, buildShipMesh, buildSailMesh, buildCannonMesh, buildHarpoonMesh, buildBoltMesh, buildBallMesh, buildWheelMesh,
  ShipEvent, CABIN, moveInCabin, gunZ, MASTS, CANNONS, HELM, HARPOON, deckZ, moveOnDeck, halfBeam, climbX, climbTop, nestZ, moveOnNest, railPoint, hullContains, HULL_CIRCLES, MAIN_DECK,
} from './ship';
import { buildAvatarParts, buildWings, avatarPose, RZ, TR, AvatarParts } from './avatar';
import { MonsterSystem, Monster, MS, ridden, segRadius, buildSegmentMesh, buildHeadMesh, BOSS_POS } from './monsters';
import { MeshBuilder } from './mesh';
import { CAVE, CAVE_SHIP, buildCave, ghostify } from './cave';
import { TENTACLES, OCTO_HP, SLAM_TIME, OctoView, buildMantleMesh, buildTentacleMesh, mantleLocal, tentaclePath } from './octopus';
import { Net, NetMsg } from './net';
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
interface Mesh { vao: WebGLVertexArrayObject; buf: WebGLBuffer; count: number; }
function vao(layout: number[], data: Float32Array | null, usage: number = G.STATIC_DRAW): Mesh {
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
const shipMesh = solid(buildShipMesh());
const sailMesh = solid(buildSailMesh());
const cannonMesh = solid(buildCannonMesh());
const harpoonMesh = solid(buildHarpoonMesh());
const boltMesh = solid(buildBoltMesh());
const ballMesh = solid(buildBallMesh());
// The ghost ship and the Leviathan's lair.
const ghostShipMesh = solid(ghostify(buildShipMesh()));
const ghostSailMesh = solid(ghostify(buildSailMesh(), true));
const ghostCannonMesh = solid(ghostify(buildCannonMesh()));
const fireLauncherMesh = solid(buildFireLauncherMesh());
const fireballMesh = solid((() => { const m = new MeshBuilder(); m.sphere([0, 0, 0], 0.9, 0.9, 0.9, 10, [2.95, 2.45, 2.1]); return m.data; })());
const eyeIsland = solid(buildEyeIsland());
const skelMeshes = (() => { const p = buildAvatarParts([0.2, 0.18, 0.15], true); return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, solid(v)])) as Record<keyof AvatarParts, Mesh>; })();
const mantleMesh = solid(buildMantleMesh()), tentMesh = solid(buildTentacleMesh());
const cave = buildCave();
const caveMesh = solid(cave.data);
const wheelMesh = solid(buildWheelMesh());
const segMesh = solid(buildSegmentMesh());
const headMesh = solid(buildHeadMesh());
const buoyMeshes = [0, 1, 2].map((k) => solid(buildBuoy(k as 0 | 1 | 2)));
const foldedWings = solid(buildWings(false));
const ropeMesh = (() => { const m = new MeshBuilder(); m.tube([0, 0, 0], [1, 0, 0], 1, 1, 6, [0.22, 0.17, 0.1]); return solid(m.data); })();
const plankMesh = (() => {
  const m = new MeshBuilder();
  m.box(0, 1, -0.45, 0.45, -0.06, 0.06, [0.4, 0.3, 0.18], [0.46, 0.35, 0.22]);
  for (let i = 0; i < 6; i++) m.box(i / 6 + 0.04, i / 6 + 0.06, -0.45, 0.45, 0.06, 0.09, [0.3, 0.22, 0.13]);
  return solid(m.data);
})();
const COATS: V3[] = [[0.55, 0.08, 0.07], [0.1, 0.2, 0.5], [0.12, 0.38, 0.16], [0.4, 0.14, 0.45]];
const COAT_CSS = ['#d9473c', '#5b8de0', '#56b46a', '#b45bd0'];
const avatarMeshes = COATS.map((c) => { const p = buildAvatarParts(c); return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, solid(v)])) as Record<keyof AvatarParts, Mesh>; });

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
const MAXP = isMobile ? 1300 : 2600;
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
const WAKE = { size: 512, res: 384, ox: 0, oy: -300 };
const mkWake = () => makeRT(WAKE.res, G.RGBA16F, G.HALF_FLOAT) ?? makeRT(WAKE.res, G.RGBA8, G.UNSIGNED_BYTE)!;
let wakeA = mkWake(), wakeB = mkWake();

// ---------- Matrix helpers ----------
function fromAxes(X: V3, Y: V3, Z: V3, p: V3): Mat4 {
  return new Float32Array([X[0], X[1], X[2], 0, Y[0], Y[1], Y[2], 0, Z[0], Z[1], Z[2], 0, p[0], p[1], p[2], 1]);
}
const SC = (x: number, y: number, z: number): Mat4 => new Float32Array([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]);
const RYm = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
const RXm = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
const mul = (...ms: Mat4[]) => ms.reduce((a, b) => multiply(a, b));
const IDENT = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist3 = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const angLerp = (a: number, b: number, t: number) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;

// ---------- World entities ----------
const audio = new SeaAudio();
const net = new Net();
const monsters = new MonsterSystem();
// One shared world; ?room=CODE opens a separate one (used for testing).
const WORLD_ROOM = (/^[A-Z0-9]{4,8}$/.exec((new URLSearchParams(location.search).get('room') ?? '').toUpperCase()) ?? ['WORLD1'])[0];
const SHIP_NAMES = ['The Black Gull', 'Sea Wraith', 'Grey Widow', 'Salt Revenant', 'Iron Kraken', 'Crimson Tide', 'Drowned Lantern', 'Storm Jackal', 'Bone Mermaid', 'Rusty Cutlass'];
interface Tether { kind: 'm' | 's'; target: number; len: number; t0: number; ap: V3; }
interface ShipEnt {
  id: number; owner: number; name: string; ship: Ship; helm: number | null; score: number;
  recoil: number[]; cannonCd: number[]; harpoonCd: number; tether: Tether | null;
  snap: { tm: number; sh: number[] } | null; inited: boolean; seen: number; remote: { rudder: number; sail: number; back?: boolean };
  marks: number[][]; preview?: boolean; wreckT: number; hitCd: number; captain: string;
  ghost?: boolean; cave?: boolean;
  groundT?: number;
  voy?: Voyage; voyNote?: { oct: boolean; skelT: number }; // the eyewall and the octopus (owner-simulated) // the ghost ship from the Leviathan's lair; cave = still down there (not broadcast)
}
interface Board { a: number; b: number; pa: [number, number]; pb: [number, number]; }
const ships = new Map<number, ShipEnt>();
let boards: Board[] = [];
// Recently cut pairs: stale snapshots still in flight must not re-lash them.
const cutAt = new Map<string, number>();
const pairKey = (a: number, b: number) => `${Math.min(a, b)}-${Math.max(a, b)}`;
const wasCut = (a: number, b: number) => now() - (cutAt.get(pairKey(a, b)) ?? -99) < 5;
let playing = false;
let time = 0;
let clockOffset: number | null = null;
const now = () => performance.now() / 1000;
const myId = () => (net.online ? net.id : 1);
const isWorldHost = () => net.role !== 'guest';
const owns = (e: ShipEnt) => e.owner === myId();

function newShipEnt(id: number, owner: number, name: string, captain: string): ShipEnt {
  return {
    id, owner, name, captain, ship: new Ship(), helm: null, score: 0, recoil: CANNONS.map(() => 0), cannonCd: CANNONS.map(() => 0), harpoonCd: 0,
    tether: null, snap: null, inited: false, seen: now(), remote: { rudder: 0, sail: 0 }, marks: [[0, 0], [0, 0], [0, 0]], wreckT: 0, hitCd: 0,
  };
}
const shipName = (id: number) => SHIP_NAMES[id % SHIP_NAMES.length];
const GHOST_NAMES = ['Drowned Bride', 'Pale Revenant', 'Wraith of the Deep', 'Hollow Crown', 'Mourning Star'];
const GHOST_SCALE = 1.45;
function makeGhost(e: ShipEnt) { e.ghost = true; e.ship.scale = GHOST_SCALE; e.ship.power = 1.7; }

type Mode = 'walk' | 'helm' | 'harpoon' | 'climb' | 'nest' | 'plank' | 'down' | 'glide' | 'ride' | 'deep' | 'land';
interface Player {
  id: number; name: string; color: number; crew: number; on: number; x: number; y: number; z: number; yaw: number; pitch: number; mode: Mode;
  hp: number; walk: number; phase: number; wave: number; aimYaw: number; aimPitch: number; moving: boolean; swing: number; mast: number;
  reel: boolean; snaps: { t: number; x: number; y: number; z: number; yaw: number; pitch: number; ay: number; ap: number; on: number }[];
  tag?: HTMLDivElement; lastStep: number; seen: number;
  // Gliding (on = -1, x/y/z in world space) and riding the Leviathan (on = RIDE_BASE - monster id).
  gv: V3; ride: number; hookMask: number; rideT: number;
  cabin: boolean; carry: boolean; // below in the cabin; carrying a flask of poison
}
const newPlayer = (id: number, name: string, color: number): Player => ({
  id, name, color, crew: 0, on: 0, x: -6, y: 0, z: MAIN_DECK, yaw: 0, pitch: 0, mode: 'walk', hp: 100, walk: 0, phase: 0, wave: 0, aimYaw: 0, aimPitch: 0,
  moving: false, swing: 0, mast: 0, reel: false, snaps: [], lastStep: 0, seen: now(), gv: [0, 0, 0], ride: 0, hookMask: 0, rideT: 0, cabin: false, carry: false,
});
const me = newPlayer(1, 'Pirate', 0);
const others = new Map<number, Player>();
let camView: 'fp' | 'tp' = 'fp';
let plankMove: { board: Board; fromA: boolean; s: number } | null = null;
let downT = 0, hurtFlash = 0, swingCd = 0;

// The ship I am standing on (for physics, camera and HUD), and my crew's ship.
const onShip = (): ShipEnt | undefined => ships.get(me.on);
const crewShip = (): ShipEnt | undefined => ships.get(me.crew);

function lockPointer() {
  if (!document.hasFocus()) return;
  try { const r = canvas.requestPointerLock() as unknown; if (r instanceof Promise) r.catch(() => undefined); } catch { /* unsupported */ }
}

const atm = { s: 0, fog: [0, 0, 0] as V3, sky: [0, 0, 0] as V3, sun: [0, 0, 0] as V3, ambT: [0, 0, 0] as V3, ambB: [0, 0, 0] as V3, fogDen: 0, rain: 0, wind: 0 };
const windVec: [number, number] = [Math.cos(WIND_DIR), Math.sin(WIND_DIR)];
function focusShip(): Ship { return onShip()?.ship ?? crewShip()?.ship ?? ships.get(0)?.ship ?? [...ships.values()][0]?.ship ?? new Ship(); }
function updateAtmosphere(dt: number) {
  if (inCave()) {
    Object.assign(atm, { s: 0, fog: [0.025, 0.06, 0.07], sky: [0.02, 0.04, 0.045], sun: [0.12, 0.2, 0.2], ambT: [0.26, 0.38, 0.4], ambB: [0.1, 0.17, 0.18], fogDen: 0.0035, rain: 0, wind: 0 });
    return;
  }
  const f = focusShip();
  atm.s += (stormAt(f.x, f.y) - atm.s) * (1 - Math.exp(-dt / 4));
  const a = 0.15 + 0.85 * atm.s;
  atm.fog = mix3([0.3, 0.34, 0.37], [0.17, 0.2, 0.23], a);
  atm.sky = mix3([0.19, 0.24, 0.3], [0.045, 0.055, 0.07], a);
  atm.sun = mix3([0.62, 0.5, 0.34], [0.15, 0.15, 0.16], a);
  atm.ambT = mix3([0.3, 0.34, 0.38], [0.24, 0.26, 0.3], a);
  atm.ambB = [atm.ambT[0] * 0.55, atm.ambT[1] * 0.55, atm.ambT[2] * 0.6];
  atm.fogDen = mix(0.00045, 0.0017, a);
  atm.rain = smoothstep(0.05, 1, a);
  atm.wind = mix(0.2, 1, a);
  // Inside the Eye: the clouds part. Blue sky and hard sunlight on still water.
  const eye = 1 - smoothstep(EYE.calm0 * 0.5, EYE.calm1, Math.hypot(f.x - EYE.x, f.y - EYE.y));
  if (eye > 0) {
    atm.fog = mix3(atm.fog, [0.6, 0.72, 0.82], eye); atm.sky = mix3(atm.sky, [0.28, 0.5, 0.82], eye);
    atm.sun = mix3(atm.sun, [1.35, 1.12, 0.85], eye); atm.ambT = mix3(atm.ambT, [0.45, 0.53, 0.62], eye);
    atm.ambB = mix3(atm.ambB, [0.3, 0.3, 0.28], eye); atm.fogDen = mix(atm.fogDen, 0.00025, eye);
  }
  visionAtmosphere();
}

const SPAWNS = { shelter: { x: 0, y: -260, psi: -Math.PI / 2, r: 120 }, storm: { x: 1900, y: -3300, psi: 2.3, r: 260 } };
function placeShip(e: ShipEnt, mode: 'shelter' | 'storm', jitter: boolean) {
  const s = SPAWNS[mode];
  let x = s.x, y = s.y;
  if (jitter) for (let k = 0; k < 20; k++) {
    const a = Math.random() * Math.PI * 2, d = Math.random() * s.r;
    x = s.x + Math.cos(a) * d; y = s.y + Math.sin(a) * d;
    if (![...ships.values()].some((o) => o !== e && !o.preview && Math.hypot(o.ship.x - x, o.ship.y - y) < 70)) break;
  }
  updatePhases(time);
  e.ship.place(x, y, s.psi);
  e.ship.sail = mode === 'storm' ? 0.6 : 0; e.ship.sailSet = mode === 'storm' ? 0.6 : 0.08; e.ship.hull = 100;
  e.inited = true; e.marks = [[x, y], [x, y], [x, y]];
}

// ---------- Input ----------
const keys = new Set<string>();
const look = { dx: 0, dy: 0 };
const stick = { active: false, id: -1, x0: 0, y0: 0, x: 0, y: 0 };
let lookPtr = -1, lookX = 0, lookY = 0;
let helmRudder = 0, helmSail = 0, helmBack = false;
let overview = { yaw: 0, pitch: 0 };
let reelHeld = false;

type Target = { kind: 'helm' | 'harpoon' | 'cannon' | 'climb' | 'plank' | 'board' | 'ashore' | 'cabin' | 'pour'; idx: number; label: string; other?: ShipEnt };
function interactTarget(): Target | null {
  if (me.mode !== 'walk') return null;
  const e = onShip();
  if (!e) return null;
  if (me.cabin) {
    if (me.x > CABIN.x1 - 0.9 && Math.abs(Math.abs(me.y) - CABIN.doorY) < CABIN.doorW + 0.3) return { kind: 'cabin', idx: 0, label: '<kbd>E</kbd> step out on deck' };
    if (Math.hypot(me.x - CABIN.chestX, Math.abs(me.y) - CABIN.chestY) < 1.7) return me.carry ? { kind: 'cabin', idx: -1, label: 'Carry the poison up to the octopus' } : { kind: 'cabin', idx: 1, label: '<kbd>E</kbd> take a flask of poison' };
    return null;
  }
  if (me.carry) { const pt = pourTarget(e); if (pt) return { kind: 'pour', idx: pt.i, label: '<kbd>E</kbd> pour the poison on the tentacle!' }; }
  if (me.x > -11 && me.x < -9.4 && Math.abs(Math.abs(me.y) - CABIN.doorY) < CABIN.doorW + 0.35) return { kind: 'cabin', idx: 2, label: '<kbd>E</kbd> enter the cabin' };
  const shore = me.x > 11 || Math.abs(me.y) > halfBeam(me.x) - 1.6 ? ashorePoint(e) : null;
  if (shore) return { kind: 'ashore', idx: 0, label: '<kbd>E</kbd> jump down onto the sand' };
  if (Math.hypot(me.x - HELM.stand[0], me.y - HELM.stand[1]) < 1.6) {
    if (me.crew !== e.id) return { kind: 'helm', idx: -1, label: `This is the ${e.name}'s wheel` };
    if (e.helm !== null && e.helm !== me.id) return { kind: 'helm', idx: -1, label: `${playerName(e.helm)} has the wheel` };
    return { kind: 'helm', idx: 0, label: '<kbd>E</kbd> take the wheel' };
  }
  if (Math.hypot(me.x - (HARPOON.x - 1.9), me.y) < 1.5) {
    const busy = [...others.values()].some((o) => o.mode === 'harpoon' && o.on === e.id);
    return busy ? { kind: 'harpoon', idx: -1, label: 'The harpoon is manned' } : { kind: 'harpoon', idx: 0, label: e.ghost ? '<kbd>E</kbd> man the fire-thrower' : '<kbd>E</kbd> man the harpoon' };
  }
  for (let i = 0; i < MASTS.length; i++) if (Math.hypot(me.x - climbX(i), me.y) < 1.2) return { kind: 'climb', idx: i, label: '<kbd>E</kbd> climb the mast' };
  // Gangplank ends on this ship.
  for (const b of boards) {
    const mine = b.a === e.id ? b.pa : b.b === e.id ? b.pb : null;
    if (mine && Math.hypot(me.x - mine[0], me.y - mine[1]) < 1.8) return { kind: 'plank', idx: boards.indexOf(b), label: '<kbd>E</kbd> cross the gangplank' };
  }
  let best = -1, bd = 1.6;
  CANNONS.forEach((c, i) => { const d = Math.hypot(me.x - c.x, me.y - (c.y - c.side * 1.5)); if (d < bd) { bd = d; best = i; } });
  if (best >= 0) return { kind: 'cannon', idx: best, label: e.cannonCd[best] <= 0 ? `<kbd>Click</kbd> fire the ${CANNONS[best].side > 0 ? 'port' : 'starboard'} cannon` : 'Reloading…' };
  // At the rail with another ship alongside: lay a gangplank.
  const side = Math.sign(me.y) || 1;
  if (Math.abs(me.y) > halfBeam(me.x) - 1.7 && me.x > -9 && me.x < 11) {
    const rp = railPoint(me.x, side), w = e.ship.local(rp[0], rp[1], MAIN_DECK);
    for (const o of ships.values()) {
      if (o === e || o.preview || boards.some((b) => (b.a === e.id && b.b === o.id) || (b.b === e.id && b.a === o.id))) continue;
      const l = toLocal(o, w);
      const ny = clamp(l[1], -halfBeam(clamp(l[0], -8.5, 10.5)), halfBeam(clamp(l[0], -8.5, 10.5)));
      if (Math.hypot(l[0] - clamp(l[0], -8.5, 10.5), l[1] - ny) < 14) return { kind: 'board', idx: 0, label: `<kbd>E</kbd> lay a gangplank to the ${o.name}`, other: o };
    }
  }
  return null;
}
function playerName(id: number) { return id === me.id ? 'You' : others.get(id)?.name ?? 'A pirate'; }
function toLocal(e: ShipEnt, w: V3): V3 {
  const { X, Y, Z } = shipAxes(e.ship.psi, e.ship.th, e.ship.ph), d = sub(w, [e.ship.x, e.ship.y, e.ship.z]);
  const k = 1 / e.ship.scale;
  return [(d[0] * X[0] + d[1] * X[1] + d[2] * X[2]) * k, (d[0] * Y[0] + d[1] * Y[1] + d[2] * Y[2]) * k, (d[0] * Z[0] + d[1] * Z[1] + d[2] * Z[2]) * k];
}

function interact() {
  if (!playing || me.mode === 'down' || me.mode === 'deep') return;
  if (me.mode === 'ride') { plantHook(1); return; }
  if (me.mode === 'glide') return;
  if (me.mode === 'land') { landInteract(); return; }
  const e = onShip();
  if (me.mode === 'helm') { leaveHelm(); return; }
  if (me.mode === 'harpoon') { me.mode = 'walk'; me.x = HARPOON.x - 1.9; me.y = 0; reelHeld = false; return; }
  if (me.mode === 'climb') {
    if (Math.abs(me.z - nestZ(me.mast)) < 1.2) { me.mode = 'nest'; me.z = nestZ(me.mast); [me.x, me.y] = moveOnNest(me.mast, me.x, me.y, -0.3, 0); me.yaw = Math.PI; toast('On the platform', 1.5); return; }
    me.mode = 'walk'; me.x = climbX(me.mast) - 0.6; me.y = 0; toast('You slide down the mast', 1.5); return;
  }
  if (me.mode === 'nest') { me.mode = 'climb'; me.x = climbX(me.mast); me.y = 0; me.z = nestZ(me.mast) + 0.2; return; }
  const t = interactTarget();
  if (!t || t.idx < 0 || !e) return;
  if (t.kind === 'helm') {
    if (owns(e)) { if (e.helm === null) { e.helm = me.id; enterHelm(); } }
    else net.to(e.owner, { t: 'claim', sid: e.id });
  } else if (t.kind === 'harpoon') { me.mode = 'harpoon'; me.aimYaw = 0; me.aimPitch = 0; }
  else if (t.kind === 'cannon') fire('c', t.idx);
  else if (t.kind === 'climb') { me.mode = 'climb'; me.mast = t.idx; me.x = climbX(t.idx); me.y = 0; me.z = MASTS[t.idx].base + 0.3; }
  else if (t.kind === 'plank') { const b = boards[t.idx]; plankMove = { board: b, fromA: b.a === e.id, s: 0 }; me.mode = 'plank'; }
  else if (t.kind === 'board' && t.other) layGangplank(e, t.other);
  else if (t.kind === 'ashore') goAshore(e);
  else if (t.kind === 'pour') { const pt = pourTarget(e); if (pt) pourPoison(e, pt.i, pt.p); }
  else if (t.kind === 'cabin') {
    if (t.idx === 2) { me.cabin = true; me.x = CABIN.x1 - 0.3; me.y = Math.sign(me.y || 1) * CABIN.doorY; me.z = MAIN_DECK; me.yaw = Math.PI; }
    else if (t.idx === 0) { me.cabin = false; me.x = -10.2; me.y = Math.sign(me.y || 1) * CABIN.doorY; me.z = deckZ(me.x, me.y); me.yaw = 0; }
    else if (t.idx === 1) { me.carry = true; audio.step(0); toast('A flask of poison. Get it onto the octopus!', 2.5); }
  }
}
function enterHelm() { me.mode = 'helm'; me.x = HELM.stand[0]; me.y = HELM.stand[1]; me.yaw = 0; me.pitch = -0.05; overview = { yaw: 0, pitch: 0 }; toast('You have the wheel', 2); }
function leaveHelm() {
  const e = onShip();
  me.mode = 'walk'; me.x = HELM.stand[0] - 0.8;
  if (!e) return;
  if (owns(e)) { if (e.helm === me.id) e.helm = null; } else net.to(e.owner, { t: 'release', sid: e.id });
}
function fireAction() {
  if (!playing || me.mode === 'down' || me.mode === 'deep') return;
  if (me.mode === 'ride') { plantHook(0); return; }
  if (me.mode === 'glide' || me.mode === 'land') return;
  const e = onShip();
  if (!e) return;
  if (me.mode === 'harpoon') {
    if (e.ghost) { fire('f', 0); return; }
    if (e.tether) { if (owns(e)) releaseTether(e, true); else net.to(e.owner, { t: 'unhook', sid: e.id }); }
    else fire('h', 0);
    return;
  }
  const t = interactTarget();
  if (t && t.kind === 'cannon') { fire('c', t.idx); return; }
  if (me.mode === 'walk' || me.mode === 'nest') slash();
}
function wave() { me.wave = 1.6; }
function toggleCam() { camView = camView === 'fp' ? 'tp' : 'fp'; }
function toggleMute() { audio.setMuted(!audio.muted); $('muteBtn').textContent = audio.muted ? 'Sound' : 'Mute'; }

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keys.add(k);
  if (!playing || e.repeat) return;
  if (k === 'e') interact();
  if (k === 'q') { if (me.mode === 'ride') plantHook(0); else wave(); }
  if (k === 'j') jump();
  if (k === 'x') sound();
  if (k === 'c') toggleCam();
  if (k === 'm') toggleMute();
  if (k === 'g') cutGangplank();
  if (k === ' ' || k === 'f') { fireAction(); e.preventDefault(); }
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));
window.addEventListener('blur', () => { keys.clear(); reelHeld = false; });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (!playing || isTouch) return;
  if (document.pointerLockElement !== canvas) { lockPointer(); if (e.button !== 0) return; }
  if (e.button === 0) fireAction();
  if (e.button === 2) { if (me.mode === 'ride') plantHook(1); else reelHeld = true; }
});
canvas.addEventListener('mouseup', (e) => { if (e.button === 2) reelHeld = false; });
document.addEventListener('mousemove', (e) => {
  if (document.pointerLockElement === canvas) { look.dx += e.movementX; look.dy += e.movementY; }
});
canvas.addEventListener('pointerdown', (e) => {
  if (!playing || e.pointerType === 'mouse') return;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  if (e.clientX < innerWidth * 0.45 && !stick.active) {
    Object.assign(stick, { active: true, id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY });
    const el = $('stick'); el.style.display = 'block'; el.style.left = `${e.clientX}px`; el.style.top = `${e.clientY}px`;
  } else if (lookPtr < 0) { lookPtr = e.pointerId; lookX = e.clientX; lookY = e.clientY; }
});
canvas.addEventListener('pointermove', (e) => {
  if (e.pointerId === stick.id && stick.active) {
    stick.x = e.clientX; stick.y = e.clientY;
    $('knob').style.transform = `translate(${clamp(stick.x - stick.x0, -50, 50)}px, ${clamp(stick.y - stick.y0, -50, 50)}px)`;
  } else if (e.pointerId === lookPtr) { look.dx += (e.clientX - lookX) * 1.6; look.dy += (e.clientY - lookY) * 1.6; lookX = e.clientX; lookY = e.clientY; }
});
const endPtr = (e: PointerEvent) => {
  if (e.pointerId === stick.id) { stick.active = false; stick.id = -1; $('stick').style.display = 'none'; $('knob').style.transform = ''; }
  if (e.pointerId === lookPtr) lookPtr = -1;
};
canvas.addEventListener('pointerup', endPtr);
canvas.addEventListener('pointercancel', endPtr);
const tap = (id: string, fn: () => void) => $(id).addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); fn(); });
tap('fireT', fireAction); tap('actT', interact); tap('waveT', wave); tap('cutT', cutGangplank);
tap('jumpT', jump); tap('diveT', sound);
$('reelT').addEventListener('pointerdown', (e) => { e.preventDefault(); reelHeld = true; });
$('reelT').addEventListener('pointerup', () => { reelHeld = false; });
$('reelT').addEventListener('pointercancel', () => { reelHeld = false; });

function stickVec(): [number, number] {
  if (!stick.active) return [0, 0];
  return [clamp((stick.x - stick.x0) / 50, -1, 1), clamp(-(stick.y - stick.y0) / 50, -1, 1)];
}

function updateControls(dt: number) {
  const sens = 0.0025;
  const [sx, sy] = stickVec();
  if (me.mode === 'harpoon') {
    me.aimYaw = clamp(me.aimYaw - look.dx * sens * 0.6, -0.85, 0.85);
    me.aimPitch = clamp(me.aimPitch - look.dy * sens * 0.6, -0.25, 0.45);
  } else if (me.mode === 'helm' && camView === 'tp') {
    overview.yaw -= look.dx * sens; overview.pitch = clamp(overview.pitch - look.dy * sens, -0.5, 0.6);
  } else { me.yaw -= look.dx * sens; me.pitch = clamp(me.pitch - look.dy * sens, -1.3, 1.3); }
  look.dx = look.dy = 0;
  const kx = (keys.has('d') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('a') || keys.has('ArrowLeft') ? 1 : 0);
  const ky = (keys.has('w') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('s') || keys.has('ArrowDown') ? 1 : 0);
  const mx = clamp(kx + sx, -1, 1), my = clamp(ky + sy, -1, 1);
  me.reel = me.mode === 'harpoon' && (reelHeld || keys.has('r'));
  const e = onShip();
  if (me.mode === 'walk' && e) {
    const run = keys.has('Shift') || Math.hypot(sx, sy) > 0.95 ? 5.5 : 3.2;
    const c = Math.cos(me.yaw), s = Math.sin(me.yaw);
    const fx = c * my + s * mx, fy = s * my - c * mx, len = Math.hypot(fx, fy);
    const slideX = -Math.sin(e.ship.th) * 2.2, slideY = -Math.sin(e.ship.ph) * 2.2;
    const vx = (len > 0.01 ? fx / Math.max(len, 1) * run : 0) + slideX + stagger.x, vy = (len > 0.01 ? fy / Math.max(len, 1) * run : 0) + slideY + stagger.y;
    if (me.cabin) { [me.x, me.y] = moveInCabin(me.x, me.y, (vx - slideX * 0.6) * dt / e.ship.scale, (vy - slideY * 0.6) * dt / e.ship.scale); me.z = MAIN_DECK; }
    else { [me.x, me.y] = moveOnDeck(me.x, me.y, vx * dt / e.ship.scale, vy * dt / e.ship.scale); me.z = deckZ(me.x, me.y); }
    me.moving = len > 0.05;
    me.walk += ((me.moving ? 1 : 0) - me.walk) * (1 - Math.exp(-dt * 8));
    if (me.moving) me.phase += dt * run * 2.4;
  } else if (me.mode === 'nest' && e) {
    // Standing on a mast platform: walk around the mast; the roll of the ship pushes you about.
    const run = 2.4, c = Math.cos(me.yaw), s = Math.sin(me.yaw);
    const fx = c * my + s * mx, fy = s * my - c * mx, len = Math.hypot(fx, fy);
    const sway = 2.4 * (me.z / 20);
    const vx = (len > 0.01 ? fx / Math.max(len, 1) * run : 0) - Math.sin(e.ship.th) * sway + stagger.x * 0.5, vy = (len > 0.01 ? fy / Math.max(len, 1) * run : 0) - Math.sin(e.ship.ph) * sway;
    [me.x, me.y] = moveOnNest(me.mast, me.x, me.y, vx * dt, vy * dt);
    me.z = nestZ(me.mast);
    me.moving = len > 0.05;
    me.walk += ((me.moving ? 1 : 0) - me.walk) * (1 - Math.exp(-dt * 8));
    if (me.moving) me.phase += dt * run * 2.4;
  } else if (me.mode === 'land') walkAshore(dt, mx, my, sx, sy);
  else if (me.mode === 'glide') updateGlide(dt, mx, my);
  else if (me.mode === 'ride') updateRide(dt, mx, my);
  else if (me.mode === 'climb') {
    const top = climbTop(me.mast);
    me.z = clamp(me.z + my * 2.6 * dt, MASTS[me.mast].base - 0.5, top);
    if (my !== 0) me.phase += dt * 6;
    if (me.z <= MASTS[me.mast].base - 0.3) { me.mode = 'walk'; me.x = climbX(me.mast) - 0.6; me.z = deckZ(me.x, 0); }
    me.walk = 0; me.moving = my !== 0;
  } else { me.walk *= Math.exp(-dt * 8); me.moving = false; }
  if (me.mode === 'helm') {
    helmRudder += clamp(-mx - helmRudder, -2 * dt, 2 * dt);
    helmSail = clamp(helmSail + my * 0.35 * dt, 0, 1);
    helmBack = my < -0.5 && helmSail < 0.03; // sails furled and still calling astern: pole her off
  }
  me.wave = Math.max(0, me.wave - dt);
  if (me.swing > 0) me.swing = Math.min(1, me.swing + dt / 0.45);
  if (me.swing >= 1) me.swing = 0;
  swingCd = Math.max(0, swingCd - dt);
}

// ---------- Sword fighting ----------
function playerWorld(p: Player): V3 | null {
  if (p.on === -1 || p.on === LAND_ON) return [p.x, p.y, p.z];
  if (p.on <= RIDE_BASE) { const m = rideMonster(RIDE_BASE - p.on); const f = m && rideFrame(m); return f ? f.pos : null; }
  const e = ships.get(p.on);
  if (!e) return null;
  return e.ship.local(p.x, p.y, p.z);
}
function slash() {
  if (swingCd > 0) return;
  swingCd = 0.6; me.swing = 0.01;
  const mw = playerWorld(me);
  const e = onShip();
  if (!mw || !e) return;
  const fwd = shipDir(e, Math.cos(me.yaw), Math.sin(me.yaw), 0);
  let hit = false;
  for (const o of others.values()) {
    if (o.crew === me.crew || o.mode === 'down') continue;
    const ow = playerWorld(o);
    if (!ow) continue;
    const d = sub(ow, mw), dist = Math.hypot(d[0], d[1], d[2] * 0.5);
    if (dist < 2.7 && (d[0] * fwd[0] + d[1] * fwd[1]) / Math.max(dist, 0.01) > 0.35) { net.to(o.id, { t: 'slash', dmg: 22 }); hit = true; }
  }
  // The drowned crew on this deck.
  for (const k of e.voy?.skel ?? []) {
    if (k.st !== 2) continue;
    const kw = e.ship.local(k.x, k.y, k.z), d = sub(kw, mw), dist = Math.hypot(d[0], d[1], d[2] * 0.5);
    if (dist < 2.7 && (d[0] * fwd[0] + d[1] * fwd[1]) / Math.max(dist, 0.01) > 0.3) {
      hit = true; burst([kw[0], kw[1], kw[2] + 1.2], 14, 3, 3, 0.6);
      if (owns(e)) hitSkeleton(e, k.id); else net.to(e.owner, { t: 'skhit', sid: e.id, k: k.id });
    }
  }
  setTimeout(() => (hit ? audio.cut(0) : audio.whoosh(0)), 130);
}
function takeDamage(dmg: number, why: string) {
  if (me.mode === 'down') return;
  me.hp -= dmg; hurtFlash = 1;
  if (me.hp <= 0) {
    me.hp = 0;
    if (me.mode === 'helm') leaveHelm();
    me.mode = 'down'; downT = 0; plankMove = null; me.carry = false;
    toast(why, 3);
  }
}
function inBoarding(): boolean { return boards.some((b) => [b.a, b.b].includes(me.crew) || [b.a, b.b].includes(me.on)); }
function revive() {
  me.cabin = false; me.carry = false;
  me.hp = 100; me.mode = 'walk'; me.on = me.crew; me.x = -6; me.y = (Math.random() - 0.5) * 3; me.z = MAIN_DECK; me.pitch = 0;
  toast('Back on your feet', 2);
}
function aliveCrew(crew: number, a: number, b: number) {
  let n = 0;
  for (const p of [me, ...others.values()]) if (p.crew === crew && p.mode !== 'down' && (p.on === a || p.on === b)) n++;
  return n;
}
function canCut(): Board | null {
  const e = onShip();
  if (!e || me.mode !== 'helm') return null;
  for (const b of boards) {
    if (b.a !== e.id && b.b !== e.id) continue;
    const other = b.a === e.id ? b.b : b.a;
    if (aliveCrew(other, b.a, b.b) === 0) return b;
  }
  return null;
}
function cutGangplank() {
  const b = canCut();
  if (!b) { if (me.mode === 'helm' && boards.some((x) => x.a === me.on || x.b === me.on)) toast('Not while their crew still stands', 2); return; }
  net.send({ t: 'unboard', a: b.a, b: b.b });
  onUnboard(b.a, b.b);
}
function layGangplank(e: ShipEnt, o: ShipEnt) {
  const side = Math.sign(me.y) || 1;
  const pa = railPoint(me.x, side);
  const w = e.ship.local(pa[0], pa[1], MAIN_DECK);
  const l = toLocal(o, w);
  const pb = railPoint(l[0], Math.sign(l[1]) || 1);
  const b: Board = { a: e.id, b: o.id, pa, pb };
  cutAt.delete(pairKey(b.a, b.b));
  boards.push(b);
  net.send({ t: 'board', a: b.a, b: b.b, pa, pb });
  toast(`Gangplank to the ${o.name}!`, 2.5);
  audio.crunch(0.5);
}
function onUnboard(a: number, b: number) {
  cutAt.set(pairKey(a, b), now());
  boards = boards.filter((x) => !((x.a === a && x.b === b) || (x.a === b && x.b === a)));
  if (me.mode === 'down' && [a, b].includes(me.crew)) revive();
  if ((me.on === a || me.on === b) && me.on !== me.crew && [a, b].includes(me.crew)) { me.on = me.crew; me.mode = 'walk'; me.x = -6; me.y = 0; me.z = MAIN_DECK; }
  if (plankMove && (plankMove.board.a === a || plankMove.board.b === a)) { plankMove = null; if (me.mode === 'plank') { me.mode = 'walk'; me.x = -6; me.y = 0; } }
  toast('The ships part ways', 2.5);
}
function updatePlank(dt: number) {
  if (me.mode !== 'plank' || !plankMove) return;
  const { board: b, fromA } = plankMove;
  const ea = ships.get(b.a), eb = ships.get(b.b);
  if (!ea || !eb) { plankMove = null; me.mode = 'walk'; return; }
  const [from, to, pf, pt] = fromA ? [ea, eb, b.pa, b.pb] : [eb, ea, b.pb, b.pa];
  plankMove.s = Math.min(1, plankMove.s + dt / 2.4);
  const P = from.ship.local(pf[0], pf[1], MAIN_DECK + 1.15), Q = to.ship.local(pt[0], pt[1], MAIN_DECK + 1.15);
  const s = plankMove.s, w: V3 = [P[0] + (Q[0] - P[0]) * s, P[1] + (Q[1] - P[1]) * s, P[2] + (Q[2] - P[2]) * s];
  const l = toLocal(from, w);
  me.on = from.id; me.x = l[0]; me.y = l[1]; me.z = l[2];
  me.walk = 1; me.moving = true; me.phase += dt * 7;
  const dirL = toLocal(from, [w[0] + (Q[0] - P[0]), w[1] + (Q[1] - P[1]), w[2]]);
  me.yaw = Math.atan2(dirL[1] - l[1], dirL[0] - l[0]);
  if (s >= 1) {
    me.on = to.id; me.mode = 'walk'; plankMove = null;
    const inward = -Math.sign(pt[1]) * 0.9;
    me.x = pt[0]; me.y = pt[1] + inward; me.z = deckZ(me.x, me.y);
    toast(to.id === me.crew ? 'Back aboard' : `Boarding the ${to.name}!`, 2);
  }
}

// ---------- Wings and Leviathan riding ----------
// Folded wings hang at each masthead: jump from the top of a mast or its platform and glide. Land
// on a deck, or on the Leviathan's head, where two hooks must be set before it shakes you off;
// with both set, the rider steers it.
const RIDE_BASE = -100;
const RIDE_GRACE = [3.5, 8]; // seconds the beast tolerates a rider with 0 or 1 hooks set
let steerTurn = 0, steerGas = 0, steerSendT = 0, remountAt = 0;
const hookCount = (mask: number) => (mask & 1) + ((mask >> 1) & 1);
const rideMonster = (id: number) => monsters.list.find((m) => m.id === id && m.st !== MS.Dying);
function rideFrame(m: Monster): { pos: V3; yaw: number } | null {
  if (m.segs.length < 3) return null;
  const a = m.segs[1], h = m.segs[0];
  return { pos: [a[0], a[1], a[2] + segRadius(1, m.scale) * 0.92], yaw: Math.atan2(h[1] - a[1], h[0] - a[0]) };
}
const aloft = () => me.mode === 'nest' || (me.mode === 'climb' && me.z > climbTop(me.mast) - 3);
const canJump = () => playing && !onShip()?.cave && (aloft() || me.mode === 'ride');
function launch(p: V3, v: V3, yaw: number) {
  me.cabin = false; me.carry = false;
  if (me.mode === 'ride') remountAt = time + 1.5; // clear the beast before you can land on it again
  me.mode = 'glide'; me.on = -1; me.x = p[0]; me.y = p[1]; me.z = p[2]; me.gv = v; me.yaw = yaw; me.pitch = -0.25;
  me.hookMask = 0; plankMove = null;
}
function jump() {
  if (!canJump()) return;
  if (me.mode === 'ride') {
    const m = rideMonster(me.ride), f = m && rideFrame(m);
    if (!m || !f) return;
    const yaw = f.yaw + me.yaw;
    launch([f.pos[0], f.pos[1], f.pos[2] + 1], [m.vx + Math.cos(yaw) * 5, m.vy + Math.sin(yaw) * 5, 9], yaw);
    toast('You leap from the beast', 1.5);
  } else {
    const e = onShip();
    if (!e) return;
    const fwd = shipDir(e, Math.cos(me.yaw), Math.sin(me.yaw), 0), yaw = Math.atan2(fwd[1], fwd[0]);
    launch(e.ship.local(me.x, me.y, me.z + 0.4), [e.ship.vx + Math.cos(yaw) * 10, e.ship.vy + Math.sin(yaw) * 10, e.ship.vz + 2.5], yaw);
    toast('Wings out!', 1.5);
  }
  audio.whoosh(0);
}
function seaRescue(msg: string, dmg: number) {
  me.cabin = false; me.carry = false;
  const w = sampleWater(me.x, me.y, camSample).z;
  burst([me.x, me.y, w], 70, 6, 9, 1.2);
  audio.splash(0.9, 0);
  me.hp = Math.max(1, me.hp - dmg);
  me.mode = 'walk'; me.on = me.crew; me.x = -6; me.y = (Math.random() - 0.5) * 3; me.z = MAIN_DECK; me.yaw = 0; me.pitch = 0; me.hookMask = 0;
  hurtFlash = 0.6;
  toast(msg, 2.5);
}
function updateGlide(dt: number, mx: number, my: number) {
  me.yaw -= mx * 1.3 * dt;
  me.pitch = clamp(me.pitch - my * 0.9 * dt, -1.2, 0.6);
  const v = me.gv, cp = Math.cos(me.pitch);
  const dir: V3 = [Math.cos(me.yaw) * cp, Math.sin(me.yaw) * cp, Math.sin(me.pitch)];
  v[2] -= 9.81 * dt;
  // The wings bend the flight path toward where you look; airspeed gives lift; drag bleeds speed.
  const sp = Math.hypot(v[0], v[1], v[2]), k = Math.min(1, 2.2 * dt);
  for (let i = 0; i < 3; i++) v[i] += (dir[i] * sp - v[i]) * k;
  const L = 9.81 * 0.95 * Math.min(1, (sp / 12) ** 2) * dt, sn = Math.sin(me.pitch);
  v[0] -= sn * Math.cos(me.yaw) * L; v[1] -= sn * Math.sin(me.yaw) * L; v[2] += cp * L; // lift, square to the flight path
  const drag = Math.exp(-dt * (0.022 + 0.0013 * sp));
  for (let i = 0; i < 3; i++) v[i] *= drag;
  me.x += v[0] * dt; me.y += v[1] * dt; me.z += v[2] * dt;
  me.walk = 0; me.moving = true;
  // Landing on a deck (any ship).
  for (const e of ships.values()) {
    if (e.preview) continue;
    const l = toLocal(e, [me.x, me.y, me.z]);
    if (Math.abs(l[0]) > 18.5 || Math.abs(l[1]) > halfBeam(clamp(l[0], -18, 18)) + 0.3) continue;
    const dz = l[2] - deckZ(l[0], l[1]);
    if (dz > 0.4 || dz < -3) continue;
    const rel = Math.hypot(v[0] - e.ship.vx, v[1] - e.ship.vy, v[2] - e.ship.vz);
    me.mode = 'walk'; me.on = e.id;
    [me.x, me.y] = moveOnDeck(l[0], l[1], 0, 0); me.z = deckZ(me.x, me.y);
    me.yaw = me.yaw - e.ship.psi; me.pitch = 0;
    cam.shake = Math.max(cam.shake, Math.min(0.4, rel * 0.015));
    audio.step(0); audio.slap(Math.min(0.5, rel * 0.03));
    if (rel > 22) takeDamage(25, 'A bone-breaking landing'); else toast(e.id === me.crew ? 'Back on deck' : `Landed on the ${e.name}!`, 2);
    return;
  }
  // Landing on the Leviathan's head.
  for (const m of monsters.list) {
    if (!m.boss || m.st === MS.Dying || time < remountAt) continue;
    for (let i = 0; i < 3 && i < m.segs.length; i++) {
      const q = m.segs[i], r = segRadius(i, m.scale);
      if (Math.hypot(me.x - q[0], me.y - q[1]) < r * 1.15 + 0.8 && me.z - q[2] > -r * 0.4 && me.z - q[2] < r + 2.5) { mount(m); return; }
    }
  }
  const ground = terrainHeight(me.x, me.y);
  if (ground > 0.3 && me.z < ground + 0.6) { seaRescue('You crashed on the rocks', 30); return; }
  if (me.z < sampleWater(me.x, me.y, camSample).z + 0.2) seaRescue('Into the sea! Fished out and hauled aboard', 10);
}
function mount(m: Monster) {
  me.mode = 'ride'; me.ride = m.id; me.on = RIDE_BASE - m.id; me.hookMask = 0; me.rideT = 0; me.yaw = 0; me.pitch = -0.15;
  steerTurn = 0; steerGas = 0;
  cam.shake = Math.max(cam.shake, 0.3);
  audio.roar(0, true); audio.hit(0);
  toast('On the Leviathan! Set both hooks!', 2.5);
}
function plantHook(side: 0 | 1) {
  if (me.mode !== 'ride' || me.hookMask & (1 << side)) return;
  me.hookMask |= 1 << side;
  audio.cut(0); audio.roar(0, true);
  cam.shake = Math.max(cam.shake, 0.15);
  if (hookCount(me.hookMask) === 2) toast('Hooks set! Steer the beast', 2.5);
}
function throwOff(msg: string) {
  const m = rideMonster(me.ride), f = m && rideFrame(m);
  const p: V3 = f ? f.pos : [me.x, me.y, me.z];
  const side = Math.random() < 0.5 ? 1 : -1, yaw = (f?.yaw ?? me.yaw) + side * Math.PI / 2;
  launch([p[0], p[1], p[2] + 1], [(m?.vx ?? 0) + Math.cos(yaw) * 8, (m?.vy ?? 0) + Math.sin(yaw) * 8, 8], yaw);
  cam.shake = Math.max(cam.shake, 0.5);
  toast(msg, 2.5);
}
function updateRide(dt: number, mx: number, my: number) {
  const m = rideMonster(me.ride), f = m && rideFrame(m);
  if (!m || !f) { throwOff('The beast is gone from under you'); return; }
  me.rideT += dt;
  me.x = f.pos[0]; me.y = f.pos[1]; me.z = f.pos[2];
  const hooks = hookCount(me.hookMask);
  if (hooks < 2 && (me.rideT > RIDE_GRACE[hooks] || (hooks === 0 && me.rideT > 0.8 && m.rear > 0.6))) { throwOff('The Leviathan shakes you off!'); return; }
  cam.shake = Math.max(cam.shake, hooks < 2 ? 0.06 : 0.015);
  if (hooks === 2) {
    steerTurn += clamp(-mx - steerTurn, -2.5 * dt, 2.5 * dt);
    steerGas += clamp(my - steerGas, -1.5 * dt, 1.5 * dt);
    steerSendT -= dt;
    if (isWorldHost()) monsters.steer(m.id, me.id, me.crew, steerTurn, steerGas);
    else if (steerSendT <= 0) { steerSendT = 0.1; net.send({ t: 'steer', m: m.id, tu: +steerTurn.toFixed(2), ga: +steerGas.toFixed(2), rs: me.crew }); }
  }
}

// ---------- Into the deep: the Leviathan's lair and the ghost ship ----------
// Flip the hooks to bare the soft underside and the Leviathan sounds, carrying its rider down to
// an air pocket in a cave. A ghost ship floats there; take her wheel and set sail, and she runs
// the drowned tunnel and breaches the surface where the beast went down.
const deep = { t: 0, from: [0, 0] as [number, number], fade: 0 };
const surge = { t: -1 };
let ghostSeq = 0, dripT = 1;
const deepEl = $('deep'), deepText = $('deepText');
const inCave = () => playing && !!onShip()?.cave;
function sound() {
  if (me.mode !== 'ride' || hookCount(me.hookMask) < 2) return;
  const m = rideMonster(me.ride);
  if (!m) return;
  me.mode = 'deep'; deep.t = 0; deep.from = [m.x, m.y];
  if (isWorldHost()) monsters.sound(m.id); else net.send({ t: 'sound', m: m.id });
  cam.shake = Math.max(cam.shake, 0.45);
  audio.roar(0, true); audio.splashdown(0.8);
  toast('Hooks flipped! It sounds into the deep…', 3);
}
function updateDeep(dt: number) {
  if (me.mode === 'deep') {
    deep.t += dt;
    const m = rideMonster(me.ride), f = m && rideFrame(m);
    if (f) { me.x = f.pos[0]; me.y = f.pos[1]; me.z = f.pos[2]; }
    cam.shake = Math.max(cam.shake, 0.08);
    if (deep.t > 5.2) enterLair();
  }
  // Fade to the deep and back out again.
  let op = 0;
  if (me.mode === 'deep') op = smoothstep(0.8, 2.6, deep.t);
  else if (surge.t >= 0) op = smoothstep(0.7, 2.4, surge.t);
  else if (deep.fade > 0) { deep.fade = Math.max(0, deep.fade - dt); op = deep.fade / 1.6; }
  deepEl.style.opacity = op.toFixed(3);
  deepEl.style.display = op > 0.005 ? 'flex' : 'none';
  deepText.style.opacity = me.mode === 'deep' ? String(smoothstep(2.2, 3.2, deep.t)) : '0';
  if (inCave() && (dripT -= dt) <= 0) { audio.drip(); dripT = 0.4 + R() * 2.2; }
}
function poseCaveShip(e: ShipEnt) {
  const sh = e.ship;
  sh.x = CAVE.x + CAVE_SHIP.x; sh.y = CAVE.y + CAVE_SHIP.y; sh.psi = CAVE_SHIP.psi;
  sh.z = 0.4 + 0.12 * Math.sin(time * 0.7); sh.th = 0.008 * Math.sin(time * 0.5); sh.ph = 0.014 * Math.sin(time * 0.6 + 1);
  sh.vx = sh.vy = sh.vz = sh.r = sh.thv = sh.phv = 0;
}
function enterLair() {
  const old = crewShip();
  const id = 100000 + myId() * 100 + (ghostSeq++ % 100);
  const e = newShipEnt(id, myId(), GHOST_NAMES[(myId() + ghostSeq) % GHOST_NAMES.length], me.name);
  makeGhost(e); e.cave = true; e.inited = true;
  ships.set(id, e);
  poseCaveShip(e); resetMarks(e);
  e.ship.sail = 0; e.ship.sailSet = 0.08;
  // Your old ship stays with her crew, or sinks if you sailed her alone.
  if (old && old.id !== id && owns(old)) {
    const mates = [...others.values()].filter((p) => p.crew === old.id).map((p) => p.id).sort((a, b) => a - b);
    if (mates.length) { old.owner = mates[0]; old.helm = null; net.send({ t: 'own', sid: old.id, to: mates[0] }); }
    else { ships.delete(old.id); boards = boards.filter((b) => b.a !== old.id && b.b !== old.id); net.send({ t: 'gone', sid: old.id }); }
  }
  me.crew = id; me.on = id; me.mode = 'walk'; me.x = -5; me.y = 0; me.z = MAIN_DECK; me.yaw = 0; me.pitch = 0.15; me.hookMask = 0;
  helmSail = 0; helmRudder = 0;
  deep.fade = 1.6; crewPanelKey = '';
  document.body.classList.add('cave');
  toast('The Leviathan\'s lair. A ghost ship waits in the dark…', 4.5);
}
function updateCaveShip(e: ShipEnt, dt: number) {
  const sh = e.ship;
  if (e.helm === myId() && me.on === e.id && me.mode === 'helm') { sh.rudder = helmRudder; sh.sail = helmSail; }
  if (e.helm === myId() && (me.mode !== 'helm' || me.on !== e.id)) e.helm = null;
  sh.sailSet += clamp(Math.max(sh.sail, 0.08) - sh.sailSet, -0.5 * dt, 0.5 * dt);
  if (surge.t < 0) {
    poseCaveShip(e);
    if (sh.sail > 0.2) { surge.t = 0; toast('Down the drowned tunnel!', 2); audio.roar(0); audio.splashdown(0.7); }
    return;
  }
  // The tunnel run: she gathers speed toward the mouth and noses down into the flooded bore.
  surge.t += dt;
  const v = Math.min(32, 3 + surge.t * 15);
  sh.vx = v; sh.x += v * dt;
  const past = sh.x - (CAVE.x + CAVE.R * 1.15);
  sh.z = 0.4 - Math.max(0, past) * 0.35; sh.th = past > 0 ? -0.2 : -0.03;
  cam.shake = Math.max(cam.shake, 0.06 + surge.t * 0.05);
  if (R() < dt * 30) emitLocal(e, [16 + R() * 3, (R() - 0.5) * 6, 0.8], [3 + R() * 5, (R() - 0.5) * 8, 3 + R() * 6], 1);
  if (surge.t > 2.5) surface(e);
}
function surface(e: ShipEnt) {
  const a = R() * Math.PI * 2, dd = 70 + R() * 80;
  const x = deep.from[0] + Math.cos(a) * dd, y = deep.from[1] + Math.sin(a) * dd;
  e.cave = false; surge.t = -1;
  updatePhases(time);
  e.ship.place(x, y, R() * Math.PI * 2);
  // She bursts up bow-first out of the deep.
  e.ship.z -= 15; e.ship.vz = 16; e.ship.th = 0.45; e.ship.sail = helmSail; e.ship.sailSet = Math.max(helmSail, 0.08);
  resetMarks(e); WAKE.ox = Math.round(x); WAKE.oy = Math.round(y); gatePrev = [x, y]; trail.length = 0;
  const w = sampleWater(x, y, camSample).z;
  burst([x, y, w], 500, 22, 26, 2);
  for (let k = 0; k < 160; k++) emitLocal(e, [-16 + R() * 34, (R() - 0.5) * 12, 1], [(R() - 0.5) * 10, (R() - 0.5) * 14, 10 + R() * 18], 2);
  audio.splashdown(1); audio.roar(0, true);
  cam.shake = Math.max(cam.shake, 0.7); wakeBlob = 1;
  deep.fade = 1.6;
  document.body.classList.remove('cave');
  toast(`The ${e.name} rises from the deep!`, 4); // names read as 'the Drowned Bride'
}
function renderCave(vp: Mat4) {
  G.enable(G.DEPTH_TEST); G.depthMask(true);
  common(pSolid.use()).m4('uVP', vp);
  draw(caveMesh, TR(CAVE.x, CAVE.y, 0));
  for (const e of ships.values()) if (e.cave) drawShip(e);
  G.enable(G.BLEND); G.depthMask(false);
  G.blendFunc(G.SRC_ALPHA, G.ONE_MINUS_SRC_ALPHA);
  common(pSpray.use()).m4('uVP', vp).f('uPx', canvas.height / Math.tan(cam.fov / 2));
  G.bindVertexArray(sprayBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, sprayBuf.buf); G.bufferData(G.ARRAY_BUFFER, sprayPos, G.DYNAMIC_DRAW);
  G.drawArrays(G.POINTS, 0, MAXP);
  G.blendFunc(G.ONE, G.ONE);
  glowCount = 0;
  for (const c of cave.crystals) {
    const p: V3 = [CAVE.x + c.p[0], CAVE.y + c.p[1], c.p[2]], d = dist3(p, cam.eye);
    addGlow(p, 3 + d * 0.01, c.col, 0.55 + 0.15 * Math.sin(time * 0.8 + c.p[0]));
  }
  for (const e of ships.values()) if (e.cave) for (const y of [-2.2, 2.2]) { const p = e.ship.local(-18.4, y, 7.1); addGlow(p, 1.6 + dist3(p, cam.eye) * 0.004, [0.35, 1, 0.6], 1.4); }
  pGlow.use().m4('uVP', vp).v3('uCamRight', cam.right).v3('uCamUp', cam.up);
  G.bindVertexArray(glowBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, glowBuf.buf);
  G.bufferData(G.ARRAY_BUFFER, glowData.subarray(0, glowCount * 60), G.DYNAMIC_DRAW);
  G.drawArrays(G.TRIANGLES, 0, glowCount * 6);
  G.disable(G.BLEND); G.depthMask(true);
  G.bindVertexArray(null);
}

// ---------- Ashore on the Eye's island ----------
const LAND_ON = -3;
const WADE = -1.5; // deepest water you can wade through
const SPICE = { x: EYE.x, y: EYE.y, r: 26 };
const aground = (e: ShipEnt) => e.ship.beached > 0 || (nearEye(e.ship.x, e.ship.y) && Math.hypot(e.ship.vx, e.ship.vy) < 1.5 && terrainHeight(...e.ship.local(17, 0, 0).slice(0, 2) as [number, number]) > -4.5);
// Where a jump from here would land: off the bow or over the near rail, if it's sand or shallows.
function ashorePoint(e: ShipEnt): V3 | null {
  if (!aground(e)) return null;
  const cands: V3[] = me.x > 11 ? [21, 24, 27].map((x) => e.ship.local(x, 0, 0)) : [2.5, 5].map((o) => e.ship.local(me.x, Math.sign(me.y) * (halfBeam(me.x) + o), 0));
  for (const c of cands) { const h = terrainHeight(c[0], c[1]); if (h > WADE) return [c[0], c[1], h]; }
  return null;
}
function goAshore(e: ShipEnt) {
  const p = ashorePoint(e);
  if (!p) return;
  const fwd = shipDir(e, Math.cos(me.yaw), Math.sin(me.yaw), 0);
  me.mode = 'land'; me.on = LAND_ON; me.x = p[0]; me.y = p[1]; me.z = Math.max(p[2], WADE); me.yaw = Math.atan2(fwd[1], fwd[0]); me.pitch = 0;
  cam.shake = Math.max(cam.shake, 0.15); audio.splash(0.3, 0); audio.step(0);
  toast('Ashore in the Eye of the Storm', 2.5);
}
function walkAshore(dt: number, mx: number, my: number, sx: number, sy: number) {
  const run = keys.has('Shift') || Math.hypot(sx, sy) > 0.95 ? 5.5 : 3.2;
  const c = Math.cos(me.yaw), s = Math.sin(me.yaw);
  const fx = c * my + s * mx, fy = s * my - c * mx, len = Math.hypot(fx, fy);
  if (len > 0.01) {
    const k = run * dt / Math.max(len, 1);
    const ok = (x: number, y: number) => terrainHeight(x, y) > WADE; // wade, but no swimming
    const nx = me.x + fx * k, ny = me.y + fy * k;
    if (ok(nx, ny)) { me.x = nx; me.y = ny; } else if (ok(nx, me.y)) me.x = nx; else if (ok(me.x, ny)) me.y = ny;
  }
  me.z = Math.max(terrainHeight(me.x, me.y), WADE);
  me.moving = len > 0.05;
  me.walk += ((me.moving ? 1 : 0) - me.walk) * (1 - Math.exp(-dt * 8));
  if (me.moving) { me.phase += dt * run * 2.4; if (Math.floor(me.phase / Math.PI) !== me.lastStep) { me.lastStep = Math.floor(me.phase / Math.PI); audio.step(0); } }
}
// The ship you could climb back aboard from here (any grounded ship close alongside).
function boardableShip(): ShipEnt | null {
  for (const e of ships.values()) {
    if (e.preview || e.cave || !aground(e)) continue;
    const l = toLocal(e, [me.x, me.y, me.z]);
    const lx = clamp(l[0], -17, 18.5), edge = Math.abs(l[1]) - halfBeam(lx);
    if (Math.abs(l[0] - lx) < 3 && edge < 3.5) return e;
  }
  return null;
}
function landTarget(): { kind: 'aboard' | 'spice'; label: string; e?: ShipEnt } | null {
  if (me.mode !== 'land') return null;
  if (Math.hypot(me.x - SPICE.x, me.y - SPICE.y) < SPICE.r + 2 && vision.kind < 0) return { kind: 'spice', label: '<kbd>E</kbd> taste the pirates\' sand' };
  const e = boardableShip();
  if (e) return { kind: 'aboard', label: `<kbd>E</kbd> climb aboard the ${e.name}`, e };
  return null;
}
function landInteract() {
  const t = landTarget();
  if (!t) return;
  if (t.kind === 'spice') { startVision(); return; }
  const e = t.e!;
  const l = toLocal(e, [me.x, me.y, me.z]);
  const yawL = me.yaw - e.ship.psi;
  me.mode = 'walk'; me.on = e.id;
  [me.x, me.y] = moveOnDeck(clamp(l[0], -16, 16), clamp(l[1], -4, 4), 0, 0); me.z = deckZ(me.x, me.y); me.yaw = yawL;
  if (!crewShip()) me.crew = e.id;
  audio.step(0); toast(`Back aboard the ${e.name}`, 2);
}

// ---------- The voyage to the Eye ----------
// Rings around the Eye: Leviathan waters out by the heart of the storm, the eyewall where the
// drowned crew climbs aboard, the octopus at the edge of the calm, then the Eye itself.
const VOYAGE_LEN = Math.hypot(BOSS_POS.x - EYE.x, BOSS_POS.y - EYE.y);
function stageAt(x: number, y: number): number {
  const d = Math.hypot(x - EYE.x, y - EYE.y);
  if (d < EYE.calm0) return 4;
  if (d < EYE.calm1) return 3;
  if (d < EYE.wall) return 2;
  if (d < VOYAGE_LEN + 900) return 1;
  return 0;
}
const voyageEl = $('voyage'), voyageName = $('voyageName'), voyageFill = $('voyageFill');
let lastStage = -1;
const STAGE_NAMES = ['', 'Leviathan waters', 'The eyewall · the drowned crew', 'The edge of the Eye', 'The Eye of the Storm'];
function updateVoyageHud() {
  const e = crewShip(), f = e ? e.ship : null;
  const st = playing && f ? stageAt(f.x, f.y) : 0;
  voyageEl.style.display = st > 0 ? 'block' : 'none';
  if (st > 0 && f && e) {
    const d = Math.hypot(f.x - EYE.x, f.y - EYE.y);
    voyageFill.style.width = `${clamp(1 - (d - EYE.calm0) / (VOYAGE_LEN - EYE.calm0), 0, 1) * 100}%`;
    const oct = e.voy?.oct;
    voyageName.textContent = st === 3 && oct ? 'The Great Octopus has the ship!' : STAGE_NAMES[st];
  }
  if (st !== lastStage && lastStage >= 0 && st > lastStage) {
    if (st === 1) toast('Leviathan waters', 3);
    if (st === 2) toast('The eyewall · the drowned rise', 3.5);
    if (st === 4) toast('The Eye of the Storm', 4);
  }
  lastStage = st;
}

// ---------- The eyewall and the edge of the Eye: the drowned crew and the Great Octopus ----------
// Each ship's owner runs its encounters and shares them in its snapshots. In the eyewall, skeletons
// climb the hull out of the sea and go for the crew; at the edge of the calm the octopus clamps on
// and holds the ship until it has been dosed with poison from the cabin chests enough times.
interface Skel { id: number; side: number; x: number; y: number; z: number; yaw: number; st: number; t: number; hp: number; cd: number; sw: number; phase: number; mv: number; }
interface Octo { hp: number; side: number; t: number; recoil: number[]; release: number; slamCd: number; }
interface Voyage { oct: Octo | null; octDone: boolean; skel: Skel[]; skelT: number; nextId: number; }
const newVoyage = (): Voyage => ({ oct: null, octDone: false, skel: [], skelT: 2, nextId: 1 });
const slams = new Map<number, { i: number; x: number; y: number; t: number; hit: boolean }>();
type DeckHand = { id: number; x: number; y: number };
function deckHands(e: ShipEnt): DeckHand[] {
  const out: DeckHand[] = [];
  const ok = (p: Player) => p.on === e.id && !p.cabin && p.mode !== 'down' && p.mode !== 'climb' && p.mode !== 'nest';
  if (playing && ok(me)) out.push({ id: me.id, x: me.x, y: me.y });
  for (const o of others.values()) if (ok(o)) out.push({ id: o.id, x: o.x, y: o.y });
  return out;
}
function updateVoyage(e: ShipEnt, dt: number) {
  const st = stageAt(e.ship.x, e.ship.y);
  if (st === 0) { if (e.voy && (e.voy.skel.length || e.voy.oct || e.voy.octDone)) e.voy = newVoyage(); return; }
  const v = e.voy ?? (e.voy = newVoyage());
  const hands = deckHands(e), crew = [me, ...others.values()].filter((p) => p.crew === e.id).length;
  if (st === 2) {
    v.skelT -= dt;
    const alive = v.skel.filter((k) => k.st < 3).length;
    if (v.skelT <= 0 && alive < Math.min(7, 3 + crew)) {
      v.skelT = 6 + R() * 4;
      const x = -9 + R() * 20, side = R() < 0.5 ? 1 : -1;
      v.skel.push({ id: v.nextId++, side, x, y: side * (halfBeam(x) + 0.5), z: -0.6, yaw: side > 0 ? -Math.PI / 2 : Math.PI / 2, st: 0, t: 0, hp: 3, cd: 1, sw: 0, phase: 0, mv: 0 });
    }
  }
  for (const k of v.skel) stepSkeleton(k, hands, dt);
  v.skel = v.skel.filter((k) => !(k.st === 3 && k.t > 2.5));
  if (st >= 3 && !v.octDone && !v.oct) v.oct = { hp: OCTO_HP, side: R() < 0.5 ? 1 : -1, t: 0, recoil: TENTACLES.map(() => 0), release: 0, slamCd: 3 };
  const o = v.oct;
  if (!o) return;
  o.t += dt;
  for (let i = 0; i < o.recoil.length; i++) o.recoil[i] = Math.max(0, o.recoil[i] - dt / 4);
  if (o.hp > 0) {
    // Clamped: the ship is held fast, heeled toward the beast, her hull groaning.
    const f = Math.exp(-dt * 2.5);
    e.ship.vx *= f; e.ship.vy *= f; e.ship.r *= f;
    e.ship.phv += o.side * 0.05 * dt;
    e.ship.hull = Math.max(1, e.ship.hull - 0.6 * dt);
    o.slamCd -= dt;
    if (o.slamCd <= 0 && !slams.has(e.id)) {
      o.slamCd = 2.4 + R() * 1.6;
      const tgt = hands.length ? hands[Math.floor(R() * hands.length)] : null;
      const tx = tgt ? tgt.x + (R() - 0.5) : TENTACLES[Math.floor(R() * TENTACLES.length)];
      let i = 0, best = 1e9;
      TENTACLES.forEach((x, j) => { const d = Math.abs(x - tx) + (o.recoil[j] > 0 ? 50 : 0); if (d < best) { best = d; i = j; } });
      const x = clamp(tx, -10.5, 12), lim = halfBeam(x) - 1;
      const y = clamp(tgt ? tgt.y + (R() - 0.5) : o.side * (lim - 1.5), -lim, lim);
      const msg = { t: 'oslam', sid: e.id, i, x: +x.toFixed(2), y: +y.toFixed(2) };
      net.send(msg); onSlam(msg);
    }
  } else {
    o.release += dt / 3;
    if (o.release >= 1) v.oct = null;
  }
}
function stepSkeleton(k: Skel, hands: DeckHand[], dt: number) {
  k.t += dt;
  if (k.sw > 0) { k.sw += dt / 0.45; if (k.sw >= 1) k.sw = 0; }
  const hb = halfBeam(k.x);
  if (k.st === 0) {
    // Climbing hand over hand up the hull out of the sea.
    const u = Math.min(1, k.t / 2.6);
    k.z = -0.6 + (gunZ(k.x) + 0.2 + 0.6) * u; k.y = k.side * (hb + 0.5 - 0.6 * u); k.phase += dt * 6;
    if (u >= 1) { k.st = 1; k.t = 0; }
  } else if (k.st === 1) {
    const u = Math.min(1, k.t / 0.5);
    k.y = k.side * (hb - 0.1 - 1.2 * u); k.z = gunZ(k.x) + 0.2 + (deckZ(k.x, k.y) - gunZ(k.x) - 0.2) * u + Math.sin(u * Math.PI) * 0.6;
    if (u >= 1) { k.st = 2; k.t = 0; }
  } else if (k.st === 2) {
    let tgt: DeckHand | null = null, bd = 1e9;
    for (const h of hands) { const d = Math.hypot(h.x - k.x, h.y - k.y); if (d < bd) { bd = d; tgt = h; } }
    k.cd -= dt; k.mv = 0;
    if (tgt) {
      const dx = tgt.x - k.x, dy = tgt.y - k.y;
      k.yaw = Math.atan2(dy, dx);
      if (bd > 1.3) { [k.x, k.y] = moveOnDeck(k.x, k.y, dx / bd * 2.3 * dt, dy / bd * 2.3 * dt); k.phase += dt * 5.5; k.mv = 1; }
      if (bd < 1.6 && k.cd <= 0) {
        k.cd = 1.4; k.sw = 0.01;
        if (tgt.id === me.id) takeDamage(12, 'A skeleton cut you down'); else net.to(tgt.id, { t: 'skslash', dmg: 12 });
      }
    }
    k.z = deckZ(k.x, k.y);
  }
}
function hitSkeleton(e: ShipEnt, id: number) {
  const k = e.voy?.skel.find((s) => s.id === id);
  if (!k || k.st === 3) return;
  k.hp -= 1;
  if (k.hp <= 0) { k.st = 3; k.t = 0; }
}
function onSlam(m: { sid: number; i: number; x: number; y: number }) {
  slams.set(m.sid, { i: m.i, x: m.x, y: m.y, t: 0, hit: false });
  const e = ships.get(m.sid);
  if (e) audio.roar(dist3(e.ship.local(m.x, m.y, 5), cam.eye), false);
}
function updateSlams(dt: number) {
  for (const [sid, sl] of slams) {
    sl.t += dt;
    const e = ships.get(sid);
    if (!sl.hit && sl.t >= SLAM_TIME && e) {
      sl.hit = true;
      const p = e.ship.local(sl.x, sl.y, deckZ(sl.x, sl.y) + 0.3), d = dist3(p, cam.eye);
      audio.crunch(0.8 / (1 + d / 50)); audio.slap(0.6 / (1 + d / 50));
      burst(p, 50, 7, 6, 0.9);
      if (me.on === sid) {
        cam.shake = Math.max(cam.shake, 0.35);
        if (!me.cabin && me.mode !== 'down' && Math.hypot(me.x - sl.x, me.y - sl.y) < 2.6) takeDamage(25, 'Crushed by a tentacle');
      }
    }
    if (sl.t > SLAM_TIME + 0.7) slams.delete(sid);
  }
}
// Poison: carried up from the cabin and poured on a tentacle on deck.
function octoView(e: ShipEnt): OctoView | null {
  const o = e.voy?.oct;
  if (!o) return null;
  const sl = slams.get(e.id);
  return { side: o.side, t: o.t, recoil: o.recoil, release: o.release, slam: sl ? { i: sl.i, x: sl.x, y: sl.y, p: sl.t / SLAM_TIME } : null };
}
function pourTarget(e: ShipEnt): { i: number; p: V3 } | null {
  const o = e.voy?.oct, view = octoView(e);
  if (!me.carry || me.cabin || !o || o.hp <= 0 || !view) return null;
  for (let i = 0; i < TENTACLES.length; i++) {
    if (o.recoil[i] > 0.3) continue;
    for (const q of tentaclePath(view, i, 12)) {
      if (q.p[2] < MAIN_DECK - 1) continue;
      if (Math.hypot(q.p[0] - me.x, q.p[1] - me.y, q.p[2] - (me.z + 1)) < 2.4) return { i, p: q.p };
    }
  }
  return null;
}
function pourPoison(e: ShipEnt, i: number, p: V3) {
  me.carry = false;
  burst(e.ship.local(p[0], p[1], p[2]), 40, 4, 5, 0.9);
  if (owns(e)) poisonOcto(e, i); else net.to(e.owner, { t: 'poison', sid: e.id, i });
}
function poisonOcto(e: ShipEnt, i: number) {
  const o = e.voy?.oct;
  if (!o || o.hp <= 0) return;
  o.hp -= 1; o.recoil[i] = 1;
  if (o.hp <= 0 && e.voy) e.voy.octDone = true;
  const msg = { t: 'opois', sid: e.id, i, hp: o.hp };
  net.send(msg); onPoison(msg);
}
function onPoison(m: { sid: number; i: number; hp: number }) {
  const e = ships.get(m.sid);
  if (!e) return;
  const d = dist3([e.ship.x, e.ship.y, e.ship.z], cam.eye);
  audio.roar(d, true); audio.splash(0.6, d);
  if (me.on === m.sid || me.crew === m.sid) toast(m.hp > 0 ? `The octopus writhes! ${OCTO_HP - m.hp} of ${OCTO_HP}` : 'The octopus lets go! The Eye lies ahead', m.hp > 0 ? 2 : 4);
}
// Crew members get the encounter state from the owner's snapshots.
function packVoyage(e: ShipEnt) {
  const v = e.voy;
  if (!v) return null;
  const o = v.oct;
  return {
    o: o ? [o.hp, o.side, +o.t.toFixed(2), +o.release.toFixed(2), ...o.recoil.map((r) => +r.toFixed(2))] : null, d: v.octDone ? 1 : 0,
    k: v.skel.map((k) => [k.id, +k.x.toFixed(2), +k.y.toFixed(2), +k.z.toFixed(2), +k.yaw.toFixed(2), k.st, k.hp, +k.sw.toFixed(2), +k.phase.toFixed(2), k.mv, +k.t.toFixed(2)]),
  };
}
function unpackVoyage(e: ShipEnt, m: { o: number[] | null; d: number; k: number[][] } | null) {
  if (!m) { e.voy = undefined; return; }
  const v = e.voy ?? (e.voy = newVoyage());
  v.octDone = !!m.d;
  v.oct = m.o ? { hp: m.o[0], side: m.o[1], t: m.o[2], release: m.o[3], recoil: m.o.slice(4), slamCd: 0 } : null;
  v.skel = m.k.map((a) => ({ id: a[0], side: 0, x: a[1], y: a[2], z: a[3], yaw: a[4], st: a[5], hp: a[6], sw: a[7], phase: a[8], mv: a[9], t: a[10], cd: 0 }));
}
// Toasts for whoever is aboard, owner or crew.
function noticeVoyage(e: ShipEnt) {
  if (me.on !== e.id && me.crew !== e.id) return;
  const n = e.voyNote ?? (e.voyNote = { oct: false, skelT: -99 });
  const oct = !!e.voy?.oct && (e.voy.oct.hp > 0);
  if (oct && !n.oct) toast('The Great Octopus seizes the ship! Fetch poison from the cabin under the wheel', 5);
  n.oct = oct;
  if (e.voy?.skel.some((k) => k.st === 0) && time - n.skelT > 25) { n.skelT = time; toast('The drowned crew climbs aboard! Cutlasses out!', 3.5); }
}
function drawVoyage(e: ShipEnt) {
  const v = e.voy;
  if (!v) return;
  const model = shipModel(e), inv = 1 / e.ship.scale;
  for (const k of v.skel) {
    const base = mul(model, TR(k.x, k.y, k.z), SC(inv, inv, inv), RZ(k.yaw));
    const anim = { phase: k.phase, walk: k.mv, wave: 0, hold: 0, time, look: 0, swing: k.sw, climb: k.st === 0 ? 1 : 0, down: k.st === 3 ? Math.min(1, k.t / 0.4) : 0, sword: k.st >= 1 };
    for (const [part, m] of avatarPose(base, anim)) draw(skelMeshes[part], m);
  }
  const view = octoView(e);
  if (!view) return;
  const S = e.ship.scale;
  const M = mantleLocal(view), mw = e.ship.local(M[0], M[1], M[2]);
  draw(mantleMesh, mul(TR(mw[0], mw[1], mw[2]), RZ(e.ship.psi + (view.side > 0 ? -Math.PI / 2 : Math.PI / 2)), RYm(0.25 + 0.05 * Math.sin(view.t)), SC(7 * S, 7 * S, 7 * S)));
  for (let i = 0; i < TENTACLES.length; i++) {
    const pts = tentaclePath(view, i).map((q) => ({ p: e.ship.local(q.p[0], q.p[1], q.p[2]), r: q.r * S }));
    for (let j = 0; j < pts.length - 1; j++) {
      const a = pts[j].p, b = pts[j + 1].p, dd = sub(b, a), len = Math.hypot(dd[0], dd[1], dd[2]) || 0.01;
      draw(tentMesh, mul(TR((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2), RZ(Math.atan2(dd[1], dd[0])), RYm(-Math.asin(clamp(dd[2] / len, -1, 1))), SC(len * 0.62, pts[j].r, pts[j].r * 0.85)));
    }
  }
}

// ---------- The pirates' sand: visions ----------
// A taste of the glowing sand on the island's crown and the future pours in for half a minute: one
// of three visions, taken in turn. Each recolours the world, bends time on the sea, fills the sky
// with things that are not there, and whispers.
interface VisionDef { name: string; fog: V3; sky: V3; sun: V3; amb: V3; amp: number; warp: number; words: string[]; }
const VISIONS: VisionDef[] = [
  {
    name: 'The Burning Armada', fog: [0.55, 0.2, 0.07], sky: [0.42, 0.09, 0.02], sun: [1.6, 0.7, 0.28], amb: [0.6, 0.3, 0.16], amp: 3.4, warp: 7,
    words: ['a thousand sails, burning', 'every sea you will ever cross', 'the tide runs red', 'you have already drowned here', 'they will sing of this fleet', 'fire on the water, forever', 'your ship is among them', 'the horizon is ash'],
  },
  {
    name: 'The Worm Ascendant', fog: [0.03, 0.07, 0.2], sky: [0.01, 0.015, 0.07], sun: [0.3, 0.45, 1.1], amb: [0.14, 0.22, 0.55], amp: 0.12, warp: 0.06,
    words: ['it was always beneath you', 'blue within blue', 'ride it — become it', 'the sand remembers the sea', 'it is older than the storm', 'it knows your name', 'the deep has an eye', 'you are inside it now'],
  },
  {
    name: 'Fractured Futures', fog: [0.3, 0.12, 0.4], sky: [0.1, 0.03, 0.18], sun: [1.2, 0.9, 1.4], amb: [0.45, 0.35, 0.6], amp: 1.8, warp: 0,
    words: ['which one is you?', 'all of them', 'you fall from the mast — you don\'t', 'the octopus never let go', 'your crew sleeps in the deep', 'you were never here', 'turn back — keep going', 'this has happened before'],
  },
];
const vision = { kind: -1, t: 0, dur: 30, next: Math.floor(Math.random() * 3), warp: 0, wordT: 0 };
const visionEl = $('vision'), visionTitle = $('visionTitle');
function startVision() {
  vision.kind = vision.next; vision.next = (vision.next + 1) % VISIONS.length;
  vision.t = 0; vision.warp = 0; vision.wordT = 2;
  visionTitle.textContent = VISIONS[vision.kind].name;
  audio.roar(0, true); cam.shake = Math.max(cam.shake, 0.3);
}
function visionK() {
  if (vision.kind < 0) return 0;
  return Math.min(1, vision.t / 3) * clamp((vision.dur - vision.t) / 4, 0, 1);
}
function spawnWord(w: string) {
  const el = document.createElement('div');
  el.className = 'vword';
  el.textContent = w;
  el.style.left = `${8 + R() * 70}%`; el.style.top = `${12 + R() * 66}%`;
  el.style.fontSize = `${18 + R() * 30}px`;
  el.style.setProperty('--rot', `${(R() - 0.5) * 24}deg`);
  visionEl.appendChild(el);
  setTimeout(() => el.remove(), 5200);
}
function updateVision(dt: number) {
  if (vision.kind < 0) return;
  vision.t += dt;
  const k = visionK(), V = VISIONS[vision.kind], t = vision.t;
  // Time on the sea: racing ahead, all but frozen, or stuttering back and forth.
  if (vision.kind === 2) { if (R() < dt * 2.5) vision.warp += (R() - 0.5) * 70 * k; }
  else vision.warp += dt * (V.warp - 1) * k;
  let f = '';
  if (vision.kind === 0) f = `hue-rotate(${-18 * k}deg) saturate(${1 + 0.9 * k}) contrast(${1 + 0.25 * k}) blur(${(0.4 + 0.6 * Math.max(0, Math.sin(t * 1.7))) * k}px)`;
  else if (vision.kind === 1) f = `saturate(${1 + 0.7 * k}) contrast(${1 + 0.35 * k}) brightness(${1 + 0.18 * k * Math.sin(t * 0.8)})`;
  else f = `hue-rotate(${(t * 140) % 360 * k}deg) saturate(${1 + 1.3 * k}) contrast(${1 + 0.45 * k})${R() < 0.05 * k ? ' invert(1)' : ''}`;
  canvas.style.filter = k > 0.01 ? f : '';
  visionEl.style.opacity = k.toFixed(3);
  visionEl.className = `v${vision.kind}`;
  visionTitle.style.opacity = String(smoothstep(0.5, 2, t) * clamp((7 - t) / 2, 0, 1));
  vision.wordT -= dt;
  if (vision.wordT <= 0 && k > 0.3) { vision.wordT = 1.1 + R() * 1.1; spawnWord(V.words[Math.floor(R() * V.words.length)]); }
  audio.setVision(k, vision.kind);
  if (vision.t >= vision.dur) {
    vision.kind = -1; canvas.style.filter = ''; visionEl.style.opacity = '0';
    audio.setVision(0, 0);
    toast('The visions fade… the sand is still warm in your mouth', 3.5);
  }
}
function visionAtmosphere() {
  const k = visionK();
  if (k <= 0) return;
  const V = VISIONS[vision.kind];
  let fog = V.fog, sun = V.sun;
  if (vision.kind === 2) {
    // Colour itself cannot hold still.
    const h = vision.t * 0.9;
    fog = [0.3 + 0.25 * Math.sin(h), 0.25 + 0.2 * Math.sin(h + 2.1), 0.35 + 0.25 * Math.sin(h + 4.2)];
    sun = [1 + 0.5 * Math.sin(h * 1.3), 0.9 + 0.4 * Math.sin(h * 1.3 + 2), 1 + 0.5 * Math.sin(h * 1.3 + 4)];
  }
  atm.fog = mix3(atm.fog, fog, k); atm.sky = mix3(atm.sky, V.sky, k); atm.sun = mix3(atm.sun, sun, k);
  atm.ambT = mix3(atm.ambT, V.amb, k); atm.ambB = mix3(atm.ambB, [V.amb[0] * 0.5, V.amb[1] * 0.5, V.amb[2] * 0.5], k);
  atm.fogDen = mix(atm.fogDen, vision.kind === 1 ? 0.0004 : 0.0009, k); atm.rain = atm.rain * (1 - k);
}
function visionCamera() {
  const k = visionK();
  if (k <= 0) return;
  const t = vision.t;
  cam.fov += k * (vision.kind === 2 ? 0.25 : 0.12) * Math.sin(t * (vision.kind === 2 ? 2.3 : 1.1));
  const a = k * (vision.kind === 2 ? 0.5 : 0.2) * Math.sin(t * 0.37);
  const c = Math.cos(a), s = Math.sin(a);
  const r = cam.right, u = cam.up;
  cam.right = [r[0] * c + u[0] * s, r[1] * c + u[1] * s, r[2] * c + u[2] * s];
  cam.up = [u[0] * c - r[0] * s, u[1] * c - r[1] * s, u[2] * c - r[2] * s];
  if (vision.kind === 0) flash = Math.max(flash, 0.5 * k * Math.max(0, Math.sin(t * 2.7)) ** 8);
}
const vhash = (a: number, b: number) => { const h = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return h - Math.floor(h); };
const visionRock = solid((() => {
  const m = new MeshBuilder();
  m.sphere([0, 0, 0], 1, 0.8, 0.9, 7, (z) => (z > 0.3 ? [0.62, 0.44, 0.26] : [0.4, 0.28, 0.17]));
  for (let k = 0; k < 6; k++) { const a = k * 1.05; m.tri([Math.cos(a) * 1.01, Math.sin(a) * 0.81, 0.1], [Math.cos(a + 0.2) * 1.01, Math.sin(a + 0.2) * 0.81, -0.2], [Math.cos(a + 0.1) * 1.0, Math.sin(a + 0.1) * 0.8, 0.4], [2.9, 2.4, 1.6]); }
  return m.data;
})());
// The things the sand shows, drawn in the solid pass around the dreamer.
function drawVisionScene() {
  const k = visionK();
  if (k <= 0.02) return;
  const t = vision.t, c: V3 = [me.x, me.y, me.z], rise = (1 - k) * -400;
  if (vision.kind === 0) {
    for (let j = 0; j < 28; j++) {
      const a = j * 2 * Math.PI / 28 + t * 0.03 * (j % 2 ? 1 : -1), R0 = 190 + (j % 4) * 95;
      const p: V3 = [c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0, 80 + (j % 5) * 55 + 25 * Math.sin(t * 0.3 + j) + rise];
      const roll = j % 7 === 0 ? Math.PI : ((j % 3) - 1) * 0.5 + 0.15 * Math.sin(t * 0.5 + j);
      const m = mul(TR(p[0], p[1], p[2]), RZ(a + Math.PI / 2), RXm(roll), RYm(0.15 * Math.sin(t * 0.4 + j * 2)));
      draw(j % 2 ? ghostShipMesh : shipMesh, m);
      for (const mt of MASTS) for (const y of mt.yards) draw(j % 2 ? ghostSailMesh : sailMesh, mul(m, TR(mt.x + 0.75, 0, y.z), SC(y.w, y.w, y.h)));
    }
  } else if (vision.kind === 1) {
    // A worm of the deep coiling up into the sky around the island, vast beyond reason.
    const N = 90;
    for (let i = N - 1; i >= 0; i--) {
      const u = i / N, a = u * 5 * Math.PI + t * 0.12, R0 = 300 - 170 * u, sc = 12 * (1 - 0.45 * u);
      const p: V3 = [c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0, -60 + 820 * u + rise];
      const u2 = (i + 1) / N, a2 = u2 * 5 * Math.PI + t * 0.12, R2 = 300 - 170 * u2;
      const q: V3 = [c[0] + Math.cos(a2) * R2, c[1] + Math.sin(a2) * R2, -60 + 820 * u2 + rise];
      const d = sub(q, p), yaw = Math.atan2(d[1], d[0]), pitch = Math.atan2(d[2], Math.hypot(d[0], d[1]));
      draw(segMesh, mul(TR(p[0], p[1], p[2]), RZ(yaw), RYm(-pitch), SC(sc * 2.6, sc * 2, sc * 2)));
    }
    const hp = worldHead(), toMe = sub(c, hp);
    draw(headMesh, mul(TR(hp[0], hp[1], hp[2]), RZ(Math.atan2(toMe[1], toMe[0])), RYm(-Math.atan2(toMe[2], Math.hypot(toMe[0], toMe[1]))), SC(14, 14, 14)));
  } else {
    // Echoes of yourself in every future at once, and the island coming apart into the sky.
    const meshes = avatarMeshes[me.color % avatarMeshes.length];
    for (let j = 0; j < 16; j++) {
      if (vhash(j, Math.floor(t * 5)) < 0.3) continue;
      const a = j * 0.9 + t * 0.25 * (j % 2 ? 1 : -1), R0 = 3 + (j % 5) * 3.2;
      const x = c[0] + Math.cos(a) * R0, y = c[1] + Math.sin(a) * R0, z = terrainHeight(x, y) + (j % 4 === 0 ? 3 + 2 * Math.sin(t + j) : 0);
      const pose = j % 6;
      const base = mul(TR(x, y, pose === 5 ? z + 4 : z), RZ(a + Math.PI / 2 + (j % 3) * 1.3), pose === 5 ? RYm(-0.3) : IDENT);
      const anim = { phase: t * 7 + j, walk: pose === 0 ? 1 : 0, wave: pose === 2 ? 1 : 0, hold: 0, time: t, look: 0, swing: pose === 1 ? (t * 1.7 + j * 0.3) % 1 : 0, climb: pose === 3 ? 1 : 0, down: pose === 4 ? 1 : 0, sword: pose < 2, glide: pose === 5 ? 1 : 0 };
      for (const [part, m] of avatarPose(base, anim)) draw(meshes[part], m);
    }
    for (let j = 0; j < 50; j++) {
      const a = vhash(j, 1) * Math.PI * 2, R0 = 10 + vhash(j, 2) * 150, sz = 1 + vhash(j, 3) * 6;
      const z = terrainHeight(c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0) + ((t * (2 + vhash(j, 4) * 4) + vhash(j, 5) * 80) % 90) * k;
      draw(visionRock, mul(TR(c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0, z), RZ(t * 0.6 + j), RXm(t * 0.4 + j), SC(sz, sz, sz)));
    }
  }
}
function worldHead(): V3 {
  const t = vision.t, k = visionK(), u = 1, a = u * 5 * Math.PI + t * 0.12, R0 = 130;
  return [me.x + Math.cos(a) * R0, me.y + Math.sin(a) * R0, -60 + 820 + 40 + (1 - k) * -400];
}
function visionGlows() {
  const k = visionK();
  if (k <= 0.02) return;
  const t = vision.t, c: V3 = [me.x, me.y, me.z];
  if (vision.kind === 0) {
    for (let j = 0; j < 28; j++) {
      const a = j * 2 * Math.PI / 28 + t * 0.03 * (j % 2 ? 1 : -1), R0 = 190 + (j % 4) * 95, z = 80 + (j % 5) * 55 + 25 * Math.sin(t * 0.3 + j) + (1 - k) * -400;
      for (let q = 0; q < 3; q++) addGlow([c[0] + Math.cos(a) * R0 + (q - 1) * 9, c[1] + Math.sin(a) * R0, z + 6 + q * 4], 9 + 5 * Math.sin(t * 6 + j + q), [1, 0.45 + 0.2 * q / 2, 0.12], 1.3 * k);
    }
  } else if (vision.kind === 1) {
    for (let j = 0; j < 160; j++) {
      const a = vhash(j, 7) * Math.PI * 2, el = 0.15 + vhash(j, 8) * 1.2, d = 2400;
      addGlow([c[0] + Math.cos(a) * Math.cos(el) * d, c[1] + Math.sin(a) * Math.cos(el) * d, Math.sin(el) * d], 5 + vhash(j, 9) * 6, [0.8, 0.85, 1], (0.4 + 0.6 * vhash(j, Math.floor(t * 3))) * k);
    }
    const hp = worldHead();
    for (const s of [1, -1]) addGlow([hp[0] + s * 14, hp[1] + s * 6, hp[2] - 6], 30 + 6 * Math.sin(t * 2), [0.35, 0.6, 1], 2.4 * k);
  } else {
    for (let j = 0; j < 50; j++) {
      const a = vhash(j, 1) * Math.PI * 2, R0 = 10 + vhash(j, 2) * 150;
      const z = terrainHeight(c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0) + ((t * (2 + vhash(j, 4) * 4) + vhash(j, 5) * 80) % 90) * k;
      addGlow([c[0] + Math.cos(a) * R0, c[1] + Math.sin(a) * R0, z], 1.5 + vhash(j, 3) * 3, [1, 0.6, 0.25], 0.9 * k);
    }
  }
}
// The spice itself glows and breathes, motes drifting up off the island's crown.
function spiceGlows() {
  if (Math.hypot(cam.eye[0] - SPICE.x, cam.eye[1] - SPICE.y) > 2500) return;
  for (let j = 0; j < 22; j++) {
    const a = vhash(j, 11) * Math.PI * 2, r = Math.sqrt(vhash(j, 12)) * SPICE.r, x = SPICE.x + Math.cos(a) * r, y = SPICE.y + Math.sin(a) * r;
    const z = terrainHeight(x, y) + ((time * (0.6 + vhash(j, 13)) + vhash(j, 14) * 8) % 8);
    addGlow([x, y, z], 0.8 + 0.6 * Math.sin(time * 2 + j), j % 3 ? [1, 0.55, 0.2] : [0.35, 0.6, 1], 0.9);
  }
}

// ---------- Camera ----------
const cam = { eye: [0, 0, 5] as V3, fwd: [1, 0, 0] as V3, right: [0, -1, 0] as V3, up: [0, 0, 1] as V3, fov: 1.2, near: 0.25, shake: 0 };
const camSample = newSample();
const chase = { pos: [0, -300, 30] as V3, yaw: -Math.PI / 2 };
function shipDir(e: ShipEnt, lx: number, ly: number, lz: number): V3 {
  const { X, Y, Z } = shipAxes(e.ship.psi, e.ship.th, e.ship.ph);
  return [X[0] * lx + Y[0] * ly + Z[0] * lz, X[1] * lx + Y[1] * ly + Z[1] * lz, X[2] * lx + Y[2] * ly + Z[2] * lz];
}
function setLook(e: ShipEnt | undefined, eye: V3, fwd: V3, stab: number) {
  const Z = e ? shipAxes(e.ship.psi, e.ship.th, e.ship.ph).Z : [0, 0, 1];
  const upRef = norm3([Z[0] * (1 - stab), Z[1] * (1 - stab), Z[2] * (1 - stab) + stab]);
  cam.eye = eye; cam.fwd = norm3(fwd);
  cam.right = norm3(cross(cam.fwd, upRef)); cam.up = cross(cam.right, cam.fwd);
}
function keepAboveWater(p: V3, margin: number) { const w = inCave() ? 0 : sampleWater(p[0], p[1], camSample).z; p[2] = Math.max(p[2], w + margin); }
function updateCamera(dt: number) {
  cam.shake *= Math.exp(-dt * 5);
  const jit = () => (Math.random() - 0.5) * cam.shake;
  const e = onShip();
  document.body.classList.toggle('fp', playing && camView === 'fp' && (me.mode === 'walk' || me.mode === 'climb' || me.mode === 'nest' || me.mode === 'harpoon' || me.mode === 'glide' || me.mode === 'ride'));
  if (playing && me.mode === 'land') {
    const cp = Math.cos(me.pitch), dir: V3 = [Math.cos(me.yaw) * cp, Math.sin(me.yaw) * cp, Math.sin(me.pitch)];
    const bob = Math.abs(Math.sin(me.phase)) * 0.06 * me.walk;
    if (camView === 'fp') setLook(undefined, [me.x + jit(), me.y + jit(), me.z + 1.68 + bob + jit()], dir, 1);
    else { const eye: V3 = [me.x - Math.cos(me.yaw) * 5.5, me.y - Math.sin(me.yaw) * 5.5, me.z + 2.6 - Math.sin(me.pitch) * 3]; eye[2] = Math.max(eye[2], terrainHeight(eye[0], eye[1]) + 1); keepAboveWater(eye, 1); setLook(undefined, eye, sub([me.x, me.y, me.z + 1.6], eye), 1); }
    cam.fov = 70 * Math.PI / 180; cam.near = 0.2;
    return;
  }
  if (playing && me.mode === 'glide') {
    const cp = Math.cos(me.pitch), dir: V3 = [Math.cos(me.yaw) * cp, Math.sin(me.yaw) * cp, Math.sin(me.pitch)];
    const sp = Math.hypot(me.gv[0], me.gv[1], me.gv[2]);
    let eye: V3 = [me.x, me.y, me.z + 0.3];
    if (camView === 'tp') { eye = [me.x - Math.cos(me.yaw) * 7, me.y - Math.sin(me.yaw) * 7, me.z + 2.2]; keepAboveWater(eye, 0.8); }
    const wob = Math.min(sp / 40, 1) * 0.04;
    setLook(undefined, [eye[0] + jit() + (Math.random() - 0.5) * wob, eye[1] + jit(), eye[2] + jit()], camView === 'tp' ? sub([me.x + dir[0] * 6, me.y + dir[1] * 6, me.z + dir[2] * 6], eye) : dir, 1);
    cam.fov = (72 + Math.min(sp, 40) * 0.5) * Math.PI / 180; cam.near = 0.2;
    return;
  }
  if (playing && (me.mode === 'ride' || me.mode === 'deep')) {
    const m = rideMonster(me.ride), f = m && rideFrame(m);
    if (f) {
      const yaw = f.yaw + me.yaw, cp = Math.cos(me.pitch);
      let eye: V3 = [f.pos[0], f.pos[1], f.pos[2] + 1.3];
      let fwd: V3 = [Math.cos(yaw) * cp, Math.sin(yaw) * cp, Math.sin(me.pitch)];
      if (camView === 'tp') { eye = [f.pos[0] - Math.cos(yaw) * 18, f.pos[1] - Math.sin(yaw) * 18, f.pos[2] + 8]; keepAboveWater(eye, 2); fwd = sub([f.pos[0], f.pos[1], f.pos[2] + 2], eye); }
      setLook(undefined, [eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], fwd, 1);
      cam.fov = 74 * Math.PI / 180; cam.near = 0.3;
      return;
    }
  }
  if (!playing || !e) {
    const f = focusShip();
    const a = Math.PI / 2 + 0.35 + time * 0.03, R = 70;
    cam.eye = [f.x + Math.cos(a) * R, f.y + Math.sin(a) * R, f.z + 18];
    setLook(undefined, cam.eye, sub([f.x, f.y - 20, f.z + 14], cam.eye), 1);
    cam.fov = 55 * Math.PI / 180; cam.near = 1;
    return;
  }
  const sp01 = Math.min(Math.hypot(e.ship.vx, e.ship.vy) / 8, 1);
  if (me.mode === 'down') {
    const a = time * 0.25, p = e.ship.local(me.x, me.y, me.z);
    cam.eye = [p[0] + Math.cos(a) * 6, p[1] + Math.sin(a) * 6, p[2] + 4];
    setLook(undefined, cam.eye, sub(p, cam.eye), 1); cam.fov = 60 * Math.PI / 180; cam.near = 0.3;
    return;
  }
  if (me.mode === 'harpoon') {
    const dl: V3 = [Math.cos(me.aimYaw) * Math.cos(me.aimPitch), Math.sin(me.aimYaw) * Math.cos(me.aimPitch), Math.sin(me.aimPitch)];
    const eye = e.ship.local(HARPOON.x - dl[0] * 3.2, -dl[1] * 3.2, HARPOON.z + (e.ghost ? 2.3 : 1.1) - dl[2] * 3.2);
    setLook(e, [eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], shipDir(e, dl[0], dl[1], dl[2] - 0.05), 0.3);
    cam.fov = 62 * Math.PI / 180; cam.near = 0.3;
    return;
  }
  if (me.mode === 'helm' && camView === 'tp') {
    chase.yaw = angLerp(chase.yaw, e.ship.psi, 1 - Math.exp(-dt * 1.2));
    const yaw = chase.yaw + overview.yaw, d = 75, h = 30 - overview.pitch * 40;
    const want: V3 = [e.ship.x - Math.cos(yaw) * d, e.ship.y - Math.sin(yaw) * d, e.ship.z + h];
    const k = 1 - Math.exp(-dt * 2.5);
    chase.pos = [chase.pos[0] + (want[0] - chase.pos[0]) * k, chase.pos[1] + (want[1] - chase.pos[1]) * k, chase.pos[2] + (want[2] - chase.pos[2]) * k];
    keepAboveWater(chase.pos, 4);
    cam.eye = [chase.pos[0] + jit(), chase.pos[1] + jit(), chase.pos[2] + jit()];
    setLook(undefined, cam.eye, sub([e.ship.x + Math.cos(e.ship.psi) * 10, e.ship.y + Math.sin(e.ship.psi) * 10, e.ship.z + 10], cam.eye), 1);
    cam.fov = (58 + 6 * sp01) * Math.PI / 180; cam.near = 1;
    return;
  }
  const dl: V3 = [Math.cos(me.yaw) * Math.cos(me.pitch), Math.sin(me.yaw) * Math.cos(me.pitch), Math.sin(me.pitch)];
  if (camView === 'fp' || me.mode === 'helm') {
    const bob = Math.abs(Math.sin(me.phase)) * 0.06 * me.walk;
    const eye = e.ship.local(me.x, me.y, me.z + (1.68 + bob) / e.ship.scale);
    setLook(e, [eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], shipDir(e, dl[0], dl[1], dl[2]), 0.35);
    cam.fov = (70 + 4 * sp01) * Math.PI / 180; cam.near = 0.2;
  } else {
    const back = me.cabin ? 1.6 : 5.5, up = me.cabin ? 2.1 : 2.6 - Math.sin(me.pitch) * 3;
    const eye = e.ship.local(me.x - Math.cos(me.yaw) * back, me.y - Math.sin(me.yaw) * back, me.z + up);
    keepAboveWater(eye, 1.5);
    setLook(e, [eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], sub(e.ship.local(me.x, me.y, me.z + 1.6), eye), 0.35);
    cam.fov = 66 * Math.PI / 180; cam.near = 0.3;
  }
}

// ---------- Spray ----------
let sprayNext = 0;
function emit(p: V3, v: V3, life = 1) {
  const i = sprayNext; sprayNext = (sprayNext + 1) % MAXP;
  sprayPos[i * 4] = p[0]; sprayPos[i * 4 + 1] = p[1]; sprayPos[i * 4 + 2] = p[2]; sprayPos[i * 4 + 3] = life;
  sprayVel[i * 3] = v[0]; sprayVel[i * 3 + 1] = v[1]; sprayVel[i * 3 + 2] = v[2];
}
function emitLocal(e: ShipEnt, l: V3, lv: V3, life = 1) {
  const p = e.ship.local(l[0], l[1], l[2]), v = shipDir(e, lv[0], lv[1], lv[2]);
  emit(p, [e.ship.vx + v[0], e.ship.vy + v[1], e.ship.vz + v[2]], life);
}
const R = () => Math.random();
function burst(p: V3, n: number, speed: number, up: number, life = 1) {
  for (let k = 0; k < n; k++) emit([p[0] + (R() - 0.5) * 2, p[1] + (R() - 0.5) * 2, p[2]], [(R() - 0.5) * speed, (R() - 0.5) * speed, up * (0.4 + R())], life);
}
function updateSpray(dt: number) {
  const drag = Math.exp(-dt * 0.7);
  for (let i = 0; i < MAXP; i++) {
    if (sprayPos[i * 4 + 3] <= 0) continue;
    sprayVel[i * 3 + 2] -= 9.81 * dt;
    sprayVel[i * 3] *= drag; sprayVel[i * 3 + 1] *= drag;
    sprayPos[i * 4] += sprayVel[i * 3] * dt; sprayPos[i * 4 + 1] += sprayVel[i * 3 + 1] * dt;
    const z = sprayPos[i * 4 + 2] += sprayVel[i * 3 + 2] * dt;
    let life = sprayPos[i * 4 + 3] - dt / 1.9;
    if (z < cam.eye[2] - 60) life = 0;
    sprayPos[i * 4 + 3] = life;
  }
}

// ---------- Weapons ----------
type ShotKind = 'c' | 'h' | 'f'; // cannonball, harpoon, fireball
interface Shot { id: number; sid: number; kind: ShotKind; p: V3; v: V3; life: number; skips: number; }
const shots: Shot[] = [];
let shotCounter = 1;
const shotSample = newSample();
function assist(p: V3, dir: V3, speed: number, cone: number, range: number, exclude: number): V3 | null {
  let best: { t: V3; score: number } | null = null;
  const consider = (q: V3, vx: number, vy: number, bias: number) => {
    const dx = q[0] - p[0], dy = q[1] - p[1], dh = Math.hypot(dx, dy);
    if (dh > range || dh < 5) return;
    const ang = Math.acos(clamp((dx * dir[0] + dy * dir[1]) / (dh * Math.hypot(dir[0], dir[1])), -1, 1));
    if (ang > cone) return;
    const score = ang * 40 + dh * 0.1 + bias;
    if (!best || score < best.score) { const tof = dh / speed; best = { t: [q[0] + vx * tof, q[1] + vy * tof, q[2]], score }; }
  };
  for (const m of monsters.list) {
    if (m.st === MS.Dying) continue;
    for (let i = 0; i < m.segs.length; i += 2) consider(m.segs[i], Math.cos(m.hd) * m.sp * (i === 0 ? 1 : 0.6), Math.sin(m.hd) * m.sp * (i === 0 ? 1 : 0.6), i);
  }
  for (const o of ships.values()) if (o.id !== exclude && !o.preview) for (const x of [-10, 0, 10]) consider(o.ship.local(x, 0, 2.5), o.ship.vx, o.ship.vy, 2);
  if (!best) return null;
  const b = best as { t: V3 };
  const dx = b.t[0] - p[0], dy = b.t[1] - p[1], x = Math.hypot(dx, dy), y = b.t[2] - p[2], g = 9.81, v2 = speed * speed;
  const disc = v2 * v2 - g * (g * x * x + 2 * y * v2);
  if (disc < 0) return null;
  const th = clamp(Math.atan((v2 - Math.sqrt(disc)) / (g * x)), -0.35, 0.35);
  return [dx / x * Math.cos(th), dy / x * Math.cos(th), Math.sin(th)];
}
function muzzle(e: ShipEnt, kind: ShotKind, idx: number, aimYaw: number, aimPitch: number): { p: V3; v: V3 } {
  const s = e.ship;
  if (kind === 'c') {
    const c = CANNONS[idx];
    const p = s.local(c.x, c.y + c.side * 1.8, c.z + 0.75);
    const raw = shipDir(e, 0, c.side, 0);
    const lvl: V3 = [-Math.sin(s.psi) * c.side, Math.cos(s.psi) * c.side, 0];
    const SPD = e.ghost ? 95 : 72;
    const d = assist(p, lvl, SPD, 0.5, e.ghost ? 220 : 160, e.id) ?? norm3([raw[0] * 0.35 + lvl[0] * 0.65, raw[1] * 0.35 + lvl[1] * 0.65, raw[2] * 0.35]);
    return { p, v: [d[0] * SPD + s.vx, d[1] * SPD + s.vy, d[2] * SPD + s.vz] };
  }
  if (kind === 'f') {
    // Lobbed, not aimed flat: a fireball arcs onto the water ahead and sets it burning.
    const pt = aimPitch + 0.12;
    const dl: V3 = [Math.cos(aimYaw) * Math.cos(pt), Math.sin(aimYaw) * Math.cos(pt), Math.sin(pt)];
    const p = s.local(HARPOON.x + dl[0] * 2.4, dl[1] * 2.4, HARPOON.z + 0.2 + dl[2] * 2.4), d = shipDir(e, dl[0], dl[1], dl[2]);
    return { p, v: [d[0] * 62 + s.vx, d[1] * 62 + s.vy, d[2] * 62 + s.vz] };
  }
  const dl: V3 = [Math.cos(aimYaw) * Math.cos(aimPitch), Math.sin(aimYaw) * Math.cos(aimPitch), Math.sin(aimPitch)];
  const p = s.local(HARPOON.x + dl[0] * 2.2, dl[1] * 2.2, HARPOON.z + 0.2 + dl[2] * 2.2);
  const raw = shipDir(e, dl[0], dl[1], dl[2]);
  const d = assist(p, raw, 80, 0.18, 220, e.id) ?? raw;
  return { p, v: [d[0] * 80 + s.vx, d[1] * 80 + s.vy, d[2] * 80 + s.vz] };
}
function fire(kind: ShotKind, idx: number) {
  const e = onShip();
  if (!e) return;
  if (owns(e)) hostFire(e, kind, idx, me.aimYaw, me.aimPitch);
  else net.to(e.owner, { t: 'fire', sid: e.id, k: kind, i: idx, ay: me.aimYaw, ap: me.aimPitch });
}
function hostFire(e: ShipEnt, kind: ShotKind, idx: number, ay: number, ap: number) {
  if (kind === 'c' ? e.cannonCd[idx] > 0 : (e.harpoonCd > 0 || e.tether)) return;
  if (kind === 'c') e.cannonCd[idx] = e.ghost ? 1.8 : 3; else e.harpoonCd = kind === 'f' ? 4.5 : 3.5;
  const { p, v } = muzzle(e, kind, idx, ay, ap);
  const id = myId() * 1000000 + shotCounter++;
  spawnShot(id, e.id, kind, idx, p, v);
  net.send({ t: 'shot', id, sid: e.id, k: kind, i: idx, p: p.map((x) => +x.toFixed(2)), v: v.map((x) => +x.toFixed(2)) });
}
function spawnShot(id: number, sid: number, kind: ShotKind, idx: number, p: V3, v: V3) {
  shots.push({ id, sid, kind, p: [...p] as V3, v: [...v] as V3, life: 8, skips: 0 });
  const e = ships.get(sid);
  const d = dist3(p, cam.eye);
  if (kind === 'c') {
    if (e) { e.recoil[idx] = 1; e.cannonCd[idx] = Math.max(e.cannonCd[idx], e.ghost ? 1.8 : 3); }
    audio.cannon(d);
    cam.shake = Math.max(cam.shake, 0.12 / (1 + d / 15));
    const dir = norm3(v);
    for (let k = 0; k < 40; k++) emit([p[0] + dir[0] * 0.5, p[1] + dir[1] * 0.5, p[2]], [dir[0] * (8 + R() * 12) + (R() - 0.5) * 4, dir[1] * (8 + R() * 12) + (R() - 0.5) * 4, (R() - 0.2) * 4], 0.5);
  } else if (kind === 'f') {
    if (e) e.harpoonCd = Math.max(e.harpoonCd, 4.5);
    audio.fireball(d);
    for (let k = 0; k < 30; k++) emit([p[0], p[1], p[2]], [v[0] * 0.15 + (R() - 0.5) * 5, v[1] * 0.15 + (R() - 0.5) * 5, (R() - 0.2) * 4], 0.5);
  } else { if (e) e.harpoonCd = Math.max(e.harpoonCd, 3.5); audio.harpoon(d); }
}
function removeShot(id: number) { const s = shots.find((x) => x.id === id); if (s) s.life = 0; }
function updateShots(dt: number) {
  for (const s of shots) {
    if (s.life <= 0) continue;
    s.v[2] -= 9.81 * dt;
    s.p[0] += s.v[0] * dt; s.p[1] += s.v[1] * dt; s.p[2] += s.v[2] * dt;
    s.life -= dt;
    const w = sampleWater(s.p[0], s.p[1], shotSample).z;
    // Serpents: the world host decides.
    if (isWorldHost()) {
      const m = monsters.hitTest(s.p, s.kind === 'h' ? 2.0 : 2.4) ?? (s.p[2] < w ? monsters.hitTest([s.p[0], s.p[1], w], 4.5) : null);
      if (m) {
        s.life = 0;
        const shooterGhost = !!ships.get(s.sid)?.ghost;
        const killed = s.kind === 'c' ? monsters.damage(m, shooterGhost ? 2 : 1) : s.kind === 'f' ? monsters.damage(m, 2) : false;
        if (killed) creditKill(s.sid, m);
        const msg = { t: 'mhit', id: s.id, sid: s.sid, p: s.p.map((x) => +x.toFixed(1)), m: m.id, h: s.kind === 'h' ? 1 : 0, k: killed ? 1 : 0, f: s.kind === 'f' ? 1 : 0 };
        net.send(msg); onMonsterHit(msg);
        continue;
      }
    }
    // Ships: each owner checks shots fired by other ships against its own hull.
    for (const e of ships.values()) {
      if (!owns(e) || e.id === s.sid || e.preview) continue;
      const l = toLocal(e, s.p);
      if (!hullContains(l, 0.3)) continue;
      s.life = 0;
      if (s.kind !== 'h') e.ship.hull = Math.max(0, e.ship.hull - (s.kind === 'f' ? 10 : ships.get(s.sid)?.ghost ? 12 : 5));
      const msg = { t: 'shit', id: s.id, sid: s.sid, tsid: e.id, p: s.p.map((x) => +x.toFixed(1)), l: l.map((x) => +x.toFixed(2)), k: s.kind };
      net.send(msg); onShipHit(msg);
      break;
    }
    if (s.life <= 0) continue;
    if (s.p[2] < w) {
      const vh = Math.hypot(s.v[0], s.v[1]);
      if (s.kind === 'c' && s.skips < 3 && vh > 30 && Math.abs(s.v[2]) < vh * 0.3) {
        s.skips++; s.p[2] = w + 0.05; s.v[2] = Math.abs(s.v[2]) * 0.45 + 2.5; s.v[0] *= 0.8; s.v[1] *= 0.8;
        burst([s.p[0], s.p[1], w], 18, 4, 7, 0.8);
        audio.splash(0.35, dist3(s.p, cam.eye));
        continue;
      }
      s.life = 0;
      if (s.kind === 'f') addFire([s.p[0], s.p[1], w], s.sid);
      burst([s.p[0], s.p[1], w], s.kind === 'c' ? 50 : 25, 5, 12, 1);
      audio.splash(s.kind === 'c' ? 0.9 : 0.5, dist3(s.p, cam.eye));
    }
  }
  for (let i = shots.length - 1; i >= 0; i--) if (shots[i].life <= 0) shots.splice(i, 1);
  for (const e of ships.values()) {
    for (let i = 0; i < CANNONS.length; i++) { e.cannonCd[i] = Math.max(0, e.cannonCd[i] - dt); e.recoil[i] = Math.max(0, e.recoil[i] - dt * 1.5); }
    if (!e.tether) e.harpoonCd = Math.max(0, e.harpoonCd - dt);
  }
}
function creditKill(sid: number, m: Monster) {
  const e = ships.get(sid);
  const add = m.boss ? 10 : 1;
  if (e && owns(e)) e.score += add; else if (e) net.to(e.owner, { t: 'score', sid, add });
}
interface HitMsg { id: number; sid: number; p: number[]; m: number; h: number; k: number; f?: number; }
function onMonsterHit(msg: HitMsg) {
  removeShot(msg.id);
  const p = msg.p as V3, d = dist3(p, cam.eye);
  const m = monsters.list.find((x) => x.id === msg.m);
  audio.hit(d); audio.roar(d, true);
  burst(p, 60, 10, 10, 1.2);
  if (msg.f) addFire(p, msg.sid);
  if (msg.h) {
    const e = ships.get(msg.sid);
    if (e && owns(e) && !e.tether && m) {
      const tip = e.ship.local(HARPOON.x + 2, 0, HARPOON.z + 0.3);
      e.tether = { kind: 'm', target: m.id, len: Math.max(15, Math.hypot(m.x - tip[0], m.y - tip[1])), t0: time, ap: [0, 0, 0] };
    }
    if (e) toast(m?.boss ? 'Harpooned the Leviathan!' : 'Harpooned!', 2);
  }
}
interface ShipHitMsg { id: number; sid: number; tsid: number; p: number[]; l: number[]; k: string; }
function onShipHit(msg: ShipHitMsg) {
  removeShot(msg.id);
  const p = msg.p as V3, d = dist3(p, cam.eye);
  audio.crunch(0.7 / (1 + d / 60));
  burst(p, 40, 8, 8, 1);
  if (msg.k === 'f') {
    addFire(p, msg.sid);
    if (me.on === msg.tsid) { cam.shake = Math.max(cam.shake, 0.3); toast('She\'s burning!', 2); }
  } else if (msg.k === 'c') {
    const mw = playerWorld(me);
    if (mw && me.on === msg.tsid && dist3(mw, p) < 3.5) takeDamage(35, 'A cannonball knocked you down');
    if (me.on === msg.tsid) cam.shake = Math.max(cam.shake, 0.2);
  } else {
    const e = ships.get(msg.sid);
    if (e && owns(e) && !e.tether) {
      const t = ships.get(msg.tsid);
      const tip = e.ship.local(HARPOON.x + 2, 0, HARPOON.z + 0.3);
      if (t) e.tether = { kind: 's', target: t.id, len: Math.max(12, dist3(tip, t.ship.local(msg.l[0], msg.l[1], msg.l[2]))), t0: time, ap: msg.l as V3 };
    }
    if (me.on === msg.tsid || me.on === msg.sid) toast('Harpoon line made fast!', 2);
  }
}

// ---------- Burning water ----------
// A fireball sets the sea alight where it lands. Each ship's owner burns its own hull; the world
// host burns the serpents. Everyone draws the flames.
interface Fire { x: number; y: number; t: number; dur: number; r: number; seed: number; sid: number; k: number; }
const fires: Fire[] = [];
const burnAcc = new Map<number, number>();
let crackleT = 0;
function addFire(p: V3 | number[], sid: number) {
  if (fires.some((f) => Math.hypot(f.x - p[0], f.y - p[1]) < 10 && f.t < 2)) return;
  fires.push({ x: p[0], y: p[1], t: 0, dur: 16, r: 22, seed: Math.random() * 1000, sid, k: 0 });
  if (fires.length > 8) fires.shift();
  const d = Math.hypot(p[0] - cam.eye[0], p[1] - cam.eye[1]);
  audio.fireball(d); audio.crunch(0.4 / (1 + d / 80));
  burst([p[0], p[1], p[2] ?? 0], 60, 10, 10, 1);
}
function updateFires(dt: number) {
  let near = 0;
  for (const f of fires) {
    f.t += dt;
    f.k = Math.min(1, f.t / 1.2) * clamp((f.dur - f.t) / 3, 0, 1);
    const R = f.r * (0.55 + 0.45 * Math.min(1, f.t / 1.5));
    for (const e of ships.values()) {
      if (!owns(e) || e.preview || e.cave) continue;
      if (HULL_CIRCLES.some((c) => { const A = e.ship.local(c.x, 0, 0); return Math.hypot(A[0] - f.x, A[1] - f.y) < R + c.r * e.ship.scale * 0.5; })) {
        e.ship.hull = Math.max(0, e.ship.hull - 4 * f.k * dt);
        if (R_() < dt * 20) emitLocal(e, [(R_() - 0.5) * 30, (R_() - 0.5) * 8, 1], [0, 0, 3 + R_() * 3], 1);
      }
    }
    if (isWorldHost()) for (const m of monsters.list) {
      if (m.st === MS.Dying || Math.hypot(m.x - f.x, m.y - f.y) > R + 3 * m.scale) continue;
      const b = (burnAcc.get(m.id) ?? 0) + dt * 0.8 * f.k;
      if (b >= 1) { burnAcc.set(m.id, b - 1); m.hurt = 1; if (monsters.damage(m, 1)) creditKill(f.sid, m); } else burnAcc.set(m.id, b);
    }
    // Steam rising off the burning water.
    if (R_() < dt * 12 * f.k) { const a = R_() * Math.PI * 2, rr = Math.sqrt(R_()) * R; emit([f.x + Math.cos(a) * rr, f.y + Math.sin(a) * rr, sampleWater(f.x + Math.cos(a) * rr, f.y + Math.sin(a) * rr, camSample).z + 1], [(R_() - 0.5) * 2, (R_() - 0.5) * 2, 4 + R_() * 4], 1.4); }
    near = Math.max(near, f.k / (1 + Math.hypot(f.x - cam.eye[0], f.y - cam.eye[1]) / 60));
  }
  for (let i = fires.length - 1; i >= 0; i--) if (fires[i].t >= fires[i].dur) fires.splice(i, 1);
  crackleT -= dt;
  if (near > 0.02 && crackleT <= 0) { audio.crackle(near); crackleT = 0.05 + R_() * 0.15; }
}
const R_ = () => Math.random();
const flameSample = newSample();
function drawFires() {
  for (const f of fires) {
    if (f.k <= 0.01 || Math.hypot(f.x - cam.eye[0], f.y - cam.eye[1]) > 2500) continue;
    const R = f.r * (0.55 + 0.45 * Math.min(1, f.t / 1.5));
    for (let q = 0; q < 28; q++) {
      const h1 = Math.sin(f.seed + q * 12.9898) * 43758.5453, h2 = Math.sin(f.seed + q * 78.233) * 12345.678;
      const a = (h1 - Math.floor(h1)) * Math.PI * 2, rr = Math.sqrt(h2 - Math.floor(h2)) * R;
      const x = f.x + Math.cos(a) * rr, y = f.y + Math.sin(a) * rr, w = sampleWater(x, y, flameSample).z;
      const fl = 0.6 + 0.4 * Math.sin(time * (7 + q % 5) + q * 1.7);
      const d = Math.hypot(x - cam.eye[0], y - cam.eye[1]);
      const a0 = 1.1 * f.k * Math.exp(-d * atm.fogDen * 0.4), core = 1 - rr / R * 0.5;
      // A tongue of fire: a broad base on the water, narrowing and reddening as it licks upward.
      addGlow([x, y, w + 1.2 + fl * 1.2], (2.6 + fl * 2.6) * core + d * 0.002, [1, 0.62, 0.18], a0 * fl);
      addGlow([x, y, w + 3.6 + fl * 2.4 * core], (2.0 + fl * 1.8) * core + d * 0.0015, [1, 0.36, 0.06], a0 * 0.8 * fl);
      if (q % 2 === 0) addGlow([x, y, w + 6 + fl * 3.5 * core], (1.3 + fl * 1.2) * core + d * 0.001, [0.9, 0.2, 0.04], a0 * 0.55 * fl);
    }
  }
}

// ---------- Harpoon lines ----------
function tetherEnds(e: ShipEnt): { tip: V3; end: V3 } | null {
  const t = e.tether;
  if (!t) return null;
  const tip = e.ship.local(HARPOON.x + 2, 0, HARPOON.z + 0.3);
  if (t.kind === 'm') {
    const m = monsters.list.find((x) => x.id === t.target);
    if (!m || !m.segs.length || m.st === MS.Dying) return null;
    return { tip, end: m.segs[1] ?? m.segs[0] };
  }
  const o = ships.get(t.target);
  if (!o) return null;
  return { tip, end: o.ship.local(t.ap[0], t.ap[1], t.ap[2]) };
}
let towToastT = -99;
function releaseTether(e: ShipEnt, rip: boolean) {
  const t = e.tether;
  if (!t) return;
  const ends = tetherEnds(e);
  const dmg = rip ? Math.round(3 + Math.min(time - t.t0, 10) * 0.5) : 0;
  e.tether = null; e.harpoonCd = 4;
  if (rip && dmg > 0) {
    const msg = { t: 'rip', sid: e.id, kind: t.kind, target: t.target, dmg, p: (ends?.end ?? [e.ship.x, e.ship.y, 0]).map((x) => +x.toFixed(1)) };
    net.send(msg); onRip(msg);
  }
}
interface RipMsg { sid: number; kind: string; target: number; dmg: number; p: number[]; }
function onRip(msg: RipMsg) {
  const d = dist3(msg.p as V3, cam.eye);
  audio.rip(d); burst(msg.p as V3, 80, 12, 12, 1.3);
  if (msg.kind === 'm') {
    audio.roar(d, true);
    if (isWorldHost()) { const m = monsters.list.find((x) => x.id === msg.target); if (m && monsters.damage(m, msg.dmg)) creditKill(msg.sid, m); }
    toast(`Ripped free: ${msg.dmg} damage`, 2);
  } else {
    const t = ships.get(msg.target);
    if (t && owns(t)) t.ship.hull = Math.max(0, t.ship.hull - msg.dmg * 2);
    if (me.on === msg.target) { cam.shake = Math.max(cam.shake, 0.3); toast('The harpoon tore out of our hull!', 2); }
  }
}
// A serpent on the line tows the ship by the bow: the bow swings toward it, and it can never
// end up astern (the line would foul the hull), only ahead or abeam.
const SHIP_LINE_K = 0.9; // tow-line stiffness between ships, (m/s²) per metre of stretch
const LINE_MAX_ANGLE = 1.55; // about 90° off the bow (abeam)
function swingToLine(e: ShipEnt, end: V3, taut: number, boss: boolean, dt: number) {
  const sh = e.ship;
  const rel = Math.atan2(Math.sin(Math.atan2(end[1] - sh.y, end[0] - sh.x) - sh.psi), Math.cos(Math.atan2(end[1] - sh.y, end[0] - sh.x) - sh.psi));
  if (taut > 0) {
    const rLine = clamp(rel * 0.45, -0.3, 0.3);
    sh.r += (rLine - sh.r) * Math.min(1, (boss ? 1.6 : 0.9) * (0.25 + 0.75 * taut) * dt);
  }
  const over = Math.abs(rel) - LINE_MAX_ANGLE;
  if (over > 0) { const turn = Math.sign(rel) * over * Math.min(1, 8 * dt); sh.psi += turn; sh.r += Math.sign(rel) * 0.05 * dt; }
}
function updateTethers(dt: number) {
  for (const e of ships.values()) {
    const t = e.tether;
    if (!t) continue;
    const ends = tetherEnds(e);
    if (!ends) { if (owns(e)) { e.tether = null; e.harpoonCd = 2; } continue; }
    const dist = Math.hypot(ends.end[0] - ends.tip[0], ends.end[1] - ends.tip[1]);
    if (owns(e)) {
      // Reel in while the harpooner holds the winch; the line snaps if badly overstretched.
      const reeling = (me.on === e.id && me.mode === 'harpoon' && me.reel) || [...others.values()].some((o) => o.on === e.id && o.mode === 'harpoon' && o.reel);
      if (reeling) t.len = Math.max(t.kind === 's' ? 14 : 9, t.len - (t.kind === 's' ? 7 : 5) * dt);
      const m = t.kind === 'm' ? monsters.list.find((x) => x.id === t.target) : undefined;
      // Reeling in on the Leviathan: the line bites harder and the ship is hauled after it like a sled.
      // A ridden Leviathan on the line is a team of horses: it drags the ship along at its own pace.
      const towed = !!m && m.boss && ridden(m);
      if (t.kind === 'm') e.ship.pullToward(ends.end[0], ends.end[1], ends.tip[0], ends.tip[1], t.len, m?.boss ? (towed ? 2.6 : reeling ? 1.6 : 0.9) : (reeling ? 0.5 : 0.25), dt, m?.vx ?? 0, m?.vy ?? 0, towed ? 18 : reeling ? 8 : 6, null, towed ? 5 : 3);
      if (towed && Math.hypot(e.ship.vx, e.ship.vy) > 12 && time - towToastT > 20 && me.on === e.id) { towToastT = time; toast('The Leviathan has you in tow!', 2.5); }
      else {
        // Ship to ship: each owner takes half of a stiff tow line, so the pair share the pull.
        const o = ships.get(t.target);
        e.ship.pullToward(ends.end[0], ends.end[1], ends.tip[0], ends.tip[1], t.len, SHIP_LINE_K, dt, o?.ship.vx ?? 0, o?.ship.vy ?? 0, 3.5, [HARPOON.x + 2, 0], 1.6);
      }
      if (t.kind === 'm') swingToLine(e, ends.end, dist > t.len * 0.9 ? Math.min((dist - t.len * 0.9) / 10, 1) : 0, !!m?.boss, dt);
      if (dist > t.len * 2.2 + 25) { releaseTether(e, true); toast('The line snapped!', 2); }
    }
    if (t.kind === 'm' && isWorldHost()) monsters.constrain(t.target, ends.tip[0], ends.tip[1], t.len, dt);
    if (t.kind === 's') { const o = ships.get(t.target); if (o && owns(o)) o.ship.pullToward(ends.tip[0], ends.tip[1], ends.end[0], ends.end[1], t.len, SHIP_LINE_K, dt, e.ship.vx, e.ship.vy, 3.5, [t.ap[0], t.ap[1]], 1.6); }
  }
  const reelingMe = me.mode === 'harpoon' && me.reel && !!onShip()?.tether;
  audio.reel(reelingMe);
}

// ---------- Ship interactions ----------
function updateBoardsAndCollisions(dt: number) {
  for (const e of ships.values()) {
    if (!owns(e) || e.preview) continue;
    // Gangplank lashing: keep the rail points about 7 m apart and the hulls parallel.
    for (const b of boards) {
      if (b.a !== e.id && b.b !== e.id) continue;
      const o = ships.get(b.a === e.id ? b.b : b.a);
      if (!o) continue;
      const mp = b.a === e.id ? b.pa : b.pb, op = b.a === e.id ? b.pb : b.pa;
      const P = e.ship.local(mp[0], mp[1], MAIN_DECK), Q = o.ship.local(op[0], op[1], MAIN_DECK);
      e.ship.pullToward(Q[0], Q[1], P[0], P[1], 7, 1.2, dt, o.ship.vx, o.ship.vy);
      let dpsi = Math.atan2(Math.sin(o.ship.psi - e.ship.psi), Math.cos(o.ship.psi - e.ship.psi));
      if (Math.abs(dpsi) > Math.PI / 2) dpsi -= Math.sign(dpsi) * Math.PI;
      e.ship.r += dpsi * 0.3 * dt;
      e.ship.vx += (o.ship.vx - e.ship.vx) * Math.min(1, 0.6 * dt); e.ship.vy += (o.ship.vy - e.ship.vy) * Math.min(1, 0.6 * dt);
    }
    // Hulls bump instead of passing through each other.
    e.hitCd = Math.max(0, e.hitCd - dt);
    for (const o of ships.values()) {
      if (o === e || o.preview || Math.hypot(o.ship.x - e.ship.x, o.ship.y - e.ship.y) > 45) continue;
      for (const c of HULL_CIRCLES) for (const d of HULL_CIRCLES) {
        const A = e.ship.local(c.x, 0, 0), B = o.ship.local(d.x, 0, 0);
        const dx = A[0] - B[0], dy = A[1] - B[1], dd = Math.hypot(dx, dy), min = c.r * e.ship.scale + d.r * o.ship.scale;
        if (dd >= min || dd < 1e-3) continue;
        const nx = dx / dd, ny = dy / dd;
        e.ship.x += nx * (min - dd) * 0.5; e.ship.y += ny * (min - dd) * 0.5;
        const close = -((e.ship.vx - o.ship.vx) * nx + (e.ship.vy - o.ship.vy) * ny);
        if (close > 0) { e.ship.vx += nx * close * 0.8; e.ship.vy += ny * close * 0.8; }
        if (close > 2.5 && e.hitCd <= 0) { e.hitCd = 1; e.ship.hull = Math.max(0, e.ship.hull - close * 3); e.ship.events.push({ type: 'crash', i: clamp(close / 6, 0.2, 1) }); }
      }
    }
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
      const p = pts[k], q = pts[k + 1], len = dist3(p, q);
      np.push([(p[0] + q[0]) / 2 + (R() - 0.5) * len * rough, (p[1] + q[1]) / 2 + (R() - 0.5) * len * rough, (p[2] + q[2]) / 2 + (R() - 0.5) * len * 0.15], q);
    }
    pts = np;
  }
  return pts;
}
function strike() {
  const f = focusShip();
  const away = Math.atan2(f.y - IC.y, f.x - IC.x) + (R() - 0.5) * 2.6;
  const d = mix(2600, 900, atm.s) + R() * mix(4200, 3000, atm.s);
  const x = cam.eye[0] + Math.cos(away) * d, y = cam.eye[1] + Math.sin(away) * d;
  const main = jagged([x, y, 1150], [x + (R() - 0.5) * 300, y + (R() - 0.5) * 300, 0], 7, 0.55);
  bolt.pts = [main];
  for (let b = 0; b < 3; b++) {
    const s = main[Math.floor(R() * main.length * 0.6)];
    bolt.pts.push(jagged(s, [s[0] + (R() - 0.5) * 500, s[1] + (R() - 0.5) * 500, s[2] - 200 - R() * 300], 5, 0.5));
  }
  bolt.mid = [x, y, 550]; bolt.t = 0;
  bolt.I = clamp(2.4 / (1 + d / 1500), 0.35, 1.7);
  audio.thunder(d);
}
function updateLightning(dt: number) {
  if (inCave()) { flash = 0; return; }
  nextStrike -= dt;
  if (nextStrike <= 0) { strike(); nextStrike = mix(24, 3.5, atm.s) * (0.45 + R()); }
  bolt.t += dt;
  const t = bolt.t;
  flash = bolt.I * (t < 0.05 ? 1 : t < 0.11 ? 0.25 : t < 0.19 ? 0.85 : t < 0.25 ? 0.2 : 0.55 * Math.exp(-(t - 0.25) * 7));
  const d = norm3(sub(bolt.mid, cam.eye)); flashDir[0] = d[0]; flashDir[1] = d[1]; flashDir[2] = d[2];
}

// ---------- Wake ----------
let wakeBlob = 0;
const segU = new Float32Array(48), segP = new Float32Array(24);
function wakePass(p: Program) {
  G.bindFramebuffer(G.FRAMEBUFFER, wakeB.fb); G.viewport(0, 0, WAKE.res, WAKE.res);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  p.i('uSrc', 0);
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  [wakeA, wakeB] = [wakeB, wakeA];
}
function updateWake() {
  const f = focusShip();
  const dx = Math.round(f.x - WAKE.ox), dy = Math.round(f.y - WAKE.oy);
  if (Math.abs(dx) > 400 || Math.abs(dy) > 400) { WAKE.ox = Math.round(f.x); WAKE.oy = Math.round(f.y); for (const e of ships.values()) resetMarks(e); }
  else if (Math.abs(dx) > 90 || Math.abs(dy) > 90) { wakePass(pShift.use().v2('uShift', dx / WAKE.size, dy / WAKE.size)); WAKE.ox += dx; WAKE.oy += dy; }
  const uv = (p: number[]) => [(p[0] - WAKE.ox) / WAKE.size + 0.5, (p[1] - WAKE.oy) / WAKE.size + 0.5];
  segP.fill(0);
  const near = [...ships.values()].filter((e) => Math.hypot(e.ship.x - WAKE.ox, e.ship.y - WAKE.oy) < 250).slice(0, 4);
  near.forEach((e, k) => {
    const st = e.ship.local(-18.5, 0, 0), po = e.ship.local(13, 3.6, 0), sb = e.ship.local(13, -3.6, 0);
    const speed = Math.hypot(e.ship.vx, e.ship.vy);
    const marks: [V3, number, number][] = [[st, 3.5 + speed * 0.2, 0.012 * clamp(speed / 6, 0, 1.3)], [po, 1.6, 0.012 * smoothstep(1.5, 7, speed)], [sb, 1.6, 0.012 * smoothstep(1.5, 7, speed)]];
    marks.forEach(([now, w, I], i) => {
      const prev = e.marks[i], a = uv(prev), b = uv(now), j = k * 3 + i;
      if (Math.hypot(now[0] - prev[0], now[1] - prev[1]) < 20) { segU.set([a[0], a[1], b[0], b[1]], j * 4); segP.set([w / WAKE.size, I], j * 2); }
      prev[0] = now[0]; prev[1] = now[1];
    });
  });
  const bow = uv(focusShip().local(17, 0, 0));
  wakePass(pWake.use().v2('uTexel', 1 / WAKE.res, 1 / WAKE.res).v4a('uSeg', segU).v2a('uSegP', segP).v3('uBlob', [bow[0], bow[1], 9 / WAKE.size]).f('uBlobI', wakeBlob));
  wakeBlob = 0;
}
function resetMarks(e: ShipEnt) { e.marks = [[e.ship.x, e.ship.y], [e.ship.x, e.ship.y], [e.ship.x, e.ship.y]]; }

// ---------- Buoys ----------
const buoySample = newSample(), buoyGx = newSample(), buoyGy = newSample();
const buoyModels: Mat4[] = BUOYS.map(() => IDENT);
const buoyTops: V3[] = BUOYS.map(() => [0, 0, 0]);
function updateBuoys() {
  BUOYS.forEach((b, i) => {
    if (Math.hypot(b.x - cam.eye[0], b.y - cam.eye[1]) > 2500) return;
    const z = sampleWater(b.x, b.y, buoySample).z;
    const gx = (sampleWater(b.x + 1.5, b.y, buoyGx).z - z) / 1.5, gy = (sampleWater(b.x, b.y + 1.5, buoyGy).z - z) / 1.5;
    const ax = shipAxes(b.phase, Math.atan(gx * 0.8), -Math.atan(gy * 0.8));
    const p: V3 = [b.x, b.y, z - 0.2];
    buoyModels[i] = fromAxes(ax.X, ax.Y, ax.Z, p);
    buoyTops[i] = [p[0] + ax.Z[0] * 3.7, p[1] + ax.Z[1] * 3.7, p[2] + ax.Z[2] * 3.7];
  });
}

// ---------- Glows ----------
const MAXG = 600;
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
const knotsEl = $('knots'), seaEl = $('sea'), leverFill = $('leverFill'), needle = $('rudderNeedle'), toastEl = $('toast');
const promptEl = $('prompt'), hullFill = $('hullfill'), killsEl = $('kills'), crewEl = $('crew'), tagsEl = $('tags');
const bossEl = $('boss'), bossFill = $('bossfill'), hpFill = $('hpfill'), hurtEl = $('hurt');
let toastTimer = 0;
function toast(msg: string, dur = 3.5) { toastEl.textContent = msg; toastEl.classList.add('show'); toastTimer = dur; }
const bearing = (dx: number, dy: number) => ((Math.atan2(dx, dy) * 180 / Math.PI) + 360) % 360;
const SEA_STATES = ['Sheltered water', 'Moderate swell', 'Rough sea', 'Very rough', 'High seas'];
let seaState = -1, seaCand = 0, seaCandT = 0;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function drawCompass(hdg: number) {
  const W = compass.width, Hh = compass.height;
  if (!W || !compass.clientWidth) return;
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
  const f = focusShip();
  const marker = (bx: number, by: number, col: string) => {
    let rel = ((bearing(bx - f.x, by - f.y) - hdg + 540) % 360) - 180;
    const edge = Math.abs(rel) > 58;
    rel = clamp(rel, -58, 58);
    const mx = W / 2 + rel * ppd;
    cctx.fillStyle = col; cctx.beginPath();
    if (edge) { const s = Math.sign(rel); cctx.moveTo(mx + s * 6 * dpr, Hh / 2); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 - 6 * dpr); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 + 6 * dpr); }
    else { cctx.moveTo(mx, Hh - 11 * dpr); cctx.lineTo(mx - 5 * dpr, Hh - 19 * dpr); cctx.lineTo(mx + 5 * dpr, Hh - 19 * dpr); }
    cctx.fill();
  };
  marker(LIGHTHOUSE.x, LIGHTHOUSE.y, '#e9b44c');
  marker(EYE.x, EYE.y, '#f4e2a1');
  const boss = monsters.list.find((m) => m.boss);
  if (boss) marker(boss.x, boss.y, '#e0624a');
  cctx.fillStyle = '#e4ebf0';
  cctx.fillRect(W / 2 - dpr, Hh - 14 * dpr, 2 * dpr, 14 * dpr);
}

function drawMinimap() {
  const W = minimap.width;
  if (!W || !minimap.clientWidth) return; // hidden (in the lair)
  const f = focusShip();
  const dpr = W / minimap.clientWidth, cx = W / 2, cy = W / 2, rad = W / 2 - 2 * dpr;
  const scale = rad / 1300;
  const mx = (x: number) => cx + (x - f.x) * scale, my = (y: number) => cy - (y - f.y) * scale;
  mctx.clearRect(0, 0, W, W);
  mctx.save();
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.clip();
  const g = mctx.createRadialGradient(mx(IC.x), my(IC.y), 600 * scale, mx(IC.x), my(IC.y), 3000 * scale);
  g.addColorStop(0, 'rgba(34,62,72,0.88)'); g.addColorStop(1, 'rgba(8,13,18,0.92)');
  mctx.fillStyle = g; mctx.fillRect(0, 0, W, W);
  mctx.setLineDash([4 * dpr, 4 * dpr]); mctx.strokeStyle = 'rgba(233,180,76,0.45)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(mx(IC.x), my(IC.y), 1500 * scale, 0, Math.PI * 2); mctx.stroke();
  mctx.strokeStyle = 'rgba(224,98,74,0.5)'; mctx.beginPath(); mctx.arc(mx(BOSS_POS.x), my(BOSS_POS.y), 900 * scale, 0, Math.PI * 2); mctx.stroke();
  // The voyage: the route from the heart of the storm, the eyewall, and the calm Eye with its island.
  mctx.strokeStyle = 'rgba(244,226,161,0.55)'; mctx.beginPath(); mctx.moveTo(mx(BOSS_POS.x), my(BOSS_POS.y)); mctx.lineTo(mx(EYE.x), my(EYE.y)); mctx.stroke();
  mctx.strokeStyle = 'rgba(180,200,220,0.45)'; mctx.beginPath(); mctx.arc(mx(EYE.x), my(EYE.y), EYE.wall * scale, 0, Math.PI * 2); mctx.stroke();
  mctx.setLineDash([]);
  mctx.fillStyle = 'rgba(120,190,200,0.25)'; mctx.beginPath(); mctx.arc(mx(EYE.x), my(EYE.y), EYE.calm1 * scale, 0, Math.PI * 2); mctx.fill();
  mctx.fillStyle = '#d6b27a'; mctx.beginPath(); mctx.arc(mx(EYE.x), my(EYE.y), Math.max(EYE.island * scale, 2.5 * dpr), 0, Math.PI * 2); mctx.fill();
  const sz = SHORE.size * scale;
  mctx.drawImage(islandImg, mx(IC.x + SHORE.min), my(IC.y + SHORE.min + SHORE.size), sz, sz);
  mctx.fillStyle = '#9aa196';
  for (const p of PILLARS) { mctx.beginPath(); mctx.arc(mx(p.x), my(p.y), Math.max(p.r * scale, 1.5 * dpr), 0, Math.PI * 2); mctx.fill(); }
  BUOYS.forEach((b) => { const c = BUOY_COLS[b.kind]; mctx.fillStyle = `rgb(${c.map((v) => Math.min(255, Math.round(v * 320))).join(',')})`; mctx.fillRect(mx(b.x) - dpr, my(b.y) - dpr, 2 * dpr, 2 * dpr); });
  mctx.fillStyle = `rgba(255,214,120,${Math.sin(time * 1.1) > 0.6 ? 1 : 0.55})`; mctx.beginPath(); mctx.arc(mx(LIGHTHOUSE.x), my(LIGHTHOUSE.y), 3 * dpr, 0, Math.PI * 2); mctx.fill();
  for (const m of monsters.list) if (m.st !== MS.Dying) { mctx.fillStyle = '#e0624a'; mctx.beginPath(); mctx.arc(mx(m.x), my(m.y), (m.boss ? 6 : 3) * dpr, 0, Math.PI * 2); mctx.fill(); }
  mctx.lineWidth = 1.5 * dpr;
  for (let i = 1; i < trail.length; i++) {
    mctx.strokeStyle = `rgba(200,225,235,${(i / trail.length) * 0.6})`;
    mctx.beginPath(); mctx.moveTo(mx(trail[i - 1][0]), my(trail[i - 1][1])); mctx.lineTo(mx(trail[i][0]), my(trail[i][1])); mctx.stroke();
  }
  for (const e of ships.values()) {
    if (e.preview && playing) continue;
    mctx.save(); mctx.translate(mx(e.ship.x), my(e.ship.y)); mctx.rotate(-e.ship.psi);
    mctx.fillStyle = e.id === me.on ? '#e9b44c' : '#e4ebf0';
    mctx.beginPath(); mctx.moveTo(8 * dpr, 0); mctx.lineTo(-5 * dpr, 4.5 * dpr); mctx.lineTo(-3 * dpr, 0); mctx.lineTo(-5 * dpr, -4.5 * dpr); mctx.fill();
    mctx.restore();
  }
  mctx.restore();
  mctx.strokeStyle = 'rgba(228,235,240,0.35)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.stroke();
  mctx.fillStyle = '#e4ebf0'; mctx.font = `600 ${10 * dpr}px system-ui`; mctx.textAlign = 'center'; mctx.fillText('N', cx, 13 * dpr);
}

let gatePrev: [number, number] = [0, 0];
function checkGates() {
  const f = focusShip();
  for (let i = 0; i < PILLARS.length - 1; i++) {
    const A = PILLARS[i], B = PILLARS[i + 1];
    const ex = B.x - A.x, ey = B.y - A.y;
    const s0 = ex * (gatePrev[1] - A.y) - ey * (gatePrev[0] - A.x), s1 = ex * (f.y - A.y) - ey * (f.x - A.x);
    if (s0 * s1 >= 0) continue;
    const t = ((f.x - A.x) * ex + (f.y - A.y) * ey) / (ex * ex + ey * ey);
    if (t < 0 || t > 1) continue;
    const islandSide = ex * (IC.y - A.y) - ey * (IC.x - A.x);
    const outward = (s1 > 0) !== (islandSide > 0);
    toast(i === ARCH.a ? (outward ? 'Through the arch' : 'Home through the arch') : outward ? 'Past the stacks' : 'Back inside the bay');
  }
  gatePrev = [f.x, f.y];
}

let crewPanelKey = '';
function updateCrewPanel() {
  const e = crewShip();
  const crew = [me, ...others.values()].filter((p) => p.crew === me.crew);
  const key = JSON.stringify([e?.name, e?.helm, crew.map((p) => [p.id, p.name, p.mode === 'down']), ships.size, net.online]);
  if (key === crewPanelKey) return;
  crewPanelKey = key;
  const rows: string[] = [];
  if (e) rows.push(`<div style="font-weight:600">${esc(e.name)}</div>`);
  for (const p of crew) rows.push(`<div class="who"><span class="dot" style="background:${COAT_CSS[p.color % COAT_CSS.length]}"></span>${esc(p.name)} <span class="role">${p.id === me.id ? '(you)' : ''} ${e?.helm === p.id ? '· at the wheel' : ''}${p.mode === 'down' ? '· down' : ''}</span></div>`);
  const otherShips = [...ships.values()].filter((s) => !s.preview && s.id !== me.crew).length;
  if (otherShips) rows.push(`<div class="role">${otherShips} other ship${otherShips > 1 ? 's' : ''} at sea</div>`);
  if (net.online && e) rows.push(`<button id="invite">Copy invite link</button>`);
  crewEl.innerHTML = rows.join('');
  const inv = document.getElementById('invite');
  if (inv) inv.onclick = () => { void navigator.clipboard?.writeText(`${location.origin}${location.pathname}?ship=${me.crew}${WORLD_ROOM === 'WORLD1' ? '' : `&room=${WORLD_ROOM}`}`).then(() => toast('Invite link copied', 2)); };
}

function updateHud(dt: number) {
  const f = focusShip(), e = onShip();
  const sp = Math.hypot(f.vx, f.vy);
  knotsEl.textContent = (sp * 1.944).toFixed(0);
  const hs = sigWaveHeight(f.x, f.y);
  const s = stormAt(f.x, f.y);
  const cat = s < 0.06 ? 0 : s < 0.28 ? 1 : s < 0.52 ? 2 : s < 0.78 ? 3 : 4;
  if (cat !== seaCand) { seaCand = cat; seaCandT = 0; } else seaCandT += dt;
  if (seaCand !== seaState && (seaCandT > 1.5 || seaState < 0)) { if (seaState >= 0 && playing) toast(SEA_STATES[seaCand]); seaState = seaCand; }
  seaEl.innerHTML = inCave() ? 'The Leviathan\'s lair' : stageAt(f.x, f.y) === 4 ? 'The Eye of the Storm · calm water' : `${SEA_STATES[Math.max(seaState, 0)]} · waves <b>${hs.toFixed(1)} m</b>`;
  leverFill.style.height = `${clamp(f.sail, 0, 1) * 100}%`;
  needle.style.left = `${50 - f.rudder * 45}%`;
  hullFill.style.width = `${f.hull}%`;
  hullFill.style.background = f.hull > 60 ? '#7fc7a0' : f.hull > 30 ? '#e9b44c' : '#e0624a';
  killsEl.textContent = `Hull ${Math.round(f.hull)}% · plunder ${crewShip()?.score ?? 0}`;
  hpFill.style.width = `${me.hp}%`;
  hurtFlash = Math.max(0, hurtFlash - dt * 1.5); hurtEl.style.opacity = String(hurtFlash * 0.6 + (me.mode === 'down' ? 0.35 : 0));
  drawCompass(((90 - f.psi * 180 / Math.PI) % 360 + 360) % 360);
  trailT += dt;
  if (trailT > 0.5) { trailT = 0; trail.push([f.x, f.y]); if (trail.length > 160) trail.shift(); }
  drawMinimap();
  const boss = monsters.list.find((m) => m.boss && m.st !== MS.Dying && Math.hypot(m.x - f.x, m.y - f.y) < 1300);
  bossEl.style.display = boss && playing ? 'block' : 'none';
  if (boss) bossFill.style.width = `${clamp(boss.hp / boss.maxHp, 0, 1) * 100}%`;
  const t = interactTarget();
  updateVoyageHud();
  let msg = t?.label ?? landTarget()?.label ?? (me.mode === 'land' && vision.kind < 0 ? (Math.hypot(me.x - SPICE.x, me.y - SPICE.y) < 160 ? 'The glowing sand on the crown of the island…' : '') : '');
  const cut = canCut();
  if (me.mode === 'helm') msg = (isTouch ? 'Stick: steer and set sail · Use: leave' : '<kbd>W</kbd>/<kbd>S</kbd> sails · <kbd>A</kbd>/<kbd>D</kbd> helm · <kbd>C</kbd> overview · <kbd>E</kbd> leave') + (cut ? ' · <kbd>G</kbd> cut her loose' : '');
  if (me.mode === 'harpoon') msg = e?.tether ? (isTouch ? 'Hold Reel to haul in · Fire to rip it out' : 'Hold <kbd>R</kbd> or right mouse to reel in · <kbd>Click</kbd> rip it out') : e && e.harpoonCd > 0 ? 'Reloading…' : isTouch ? 'Drag to aim · Fire' : 'Mouse to aim · <kbd>Click</kbd> fire · <kbd>E</kbd> leave';
  if (me.mode === 'harpoon' && e?.ghost) msg = e.harpoonCd > 0 ? 'Stoking the fire…' : isTouch ? 'Drag to aim · Fire: set the sea ablaze' : 'Mouse to aim · <kbd>Click</kbd> hurl fire and set the sea ablaze · <kbd>E</kbd> leave';
  if (me.mode === 'climb') {
    const atNest = Math.abs(me.z - nestZ(me.mast)) < 1.2;
    msg = isTouch ? (atNest ? 'Use: step onto the platform' : 'Stick up and down to climb · Use: slide down') : atNest ? '<kbd>E</kbd> step onto the platform · <kbd>W</kbd>/<kbd>S</kbd> climb' : '<kbd>W</kbd>/<kbd>S</kbd> climb · <kbd>E</kbd> slide down';
  }
  if (me.mode === 'nest') msg = isTouch ? 'Use: back onto the mast' : '<kbd>E</kbd> back onto the mast';
  if (aloft()) msg += isTouch ? ' · Jump: glide' : ' · <kbd>J</kbd> jump and glide';
  if (me.mode === 'glide') msg = isTouch ? 'Stick: dive, climb and turn · land on a deck or the Leviathan\'s head' : 'Mouse or <kbd>W</kbd>/<kbd>S</kbd> dive and climb · <kbd>A</kbd>/<kbd>D</kbd> turn · land on a deck or the Leviathan\'s head';
  if (me.mode === 'ride') {
    const h = me.hookMask, n = hookCount(h);
    if (n < 2) {
      const left = RIDE_GRACE[n] - me.rideT;
      msg = `Set your hooks! ${h & 1 ? '✓' : isTouch ? 'Left hook' : '<kbd>Click</kbd>/<kbd>Q</kbd> left'} · ${h & 2 ? '✓' : isTouch ? 'Right hook' : '<kbd>Right-click</kbd>/<kbd>E</kbd> right'} · ${Math.max(0, left).toFixed(1)} s`;
    } else msg = isTouch ? 'Stick: steer and drive the beast · Jump: leap off · Dive: flip the hooks' : '<kbd>A</kbd>/<kbd>D</kbd> steer · <kbd>W</kbd>/<kbd>S</kbd> faster, slower · ram their ships · <kbd>J</kbd> leap off · <kbd>X</kbd> flip the hooks and dive';
  }
  if (inCave()) {
    if (me.mode === 'helm') msg = surge.t >= 0 ? '' : isTouch ? 'Stick up: set sail and run the drowned tunnel' : '<kbd>W</kbd> set sail and run the drowned tunnel to the surface';
    else if (me.mode === 'walk' && !msg) msg = 'Take the ghost ship\'s wheel to sail out';
  }
  if (me.mode === 'down') msg = inBoarding() ? 'You are down until the boarding is settled' : `Recovering… ${Math.max(0, Math.ceil(8 - downT))}`;
  if (isTouch) msg = msg.replace(/<kbd>(Click|E|G)<\/kbd> /g, '');
  promptEl.innerHTML = msg; promptEl.classList.toggle('show', !!msg && playing);
  const riding = me.mode === 'ride', setting = riding && hookCount(me.hookMask) < 2;
  const fireLabel = riding ? 'Left hook' : me.mode === 'harpoon' ? (e?.tether ? 'Rip' : 'Fire') : t?.kind === 'cannon' ? 'Fire' : 'Slash';
  $('fireT').textContent = fireLabel;
  $('fireT').style.display = me.mode === 'walk' || me.mode === 'nest' || me.mode === 'harpoon' || (setting && !(me.hookMask & 1)) ? 'block' : 'none';
  $('actT').style.display = riding ? (setting && !(me.hookMask & 2) ? 'block' : 'none') : ((t && t.idx >= 0 && t.kind !== 'cannon') || (me.mode !== 'walk' && me.mode !== 'down' && me.mode !== 'plank' && me.mode !== 'glide' && me.mode !== 'land') || !!landTarget()) ? 'block' : 'none';
  $('diveT').style.display = riding && !setting ? 'block' : 'none';
  $('jumpT').style.display = canJump() && !setting ? 'block' : 'none';
  $('actT').textContent = me.mode === 'land' ? (landTarget()?.kind === 'spice' ? 'Taste' : 'Aboard') : me.mode === 'walk' ? ({ helm: 'Wheel', harpoon: 'Harpoon', climb: 'Climb', plank: 'Cross', board: 'Gangplank', cannon: 'Fire', ashore: 'Ashore', pour: 'Pour', cabin: t?.idx === 1 ? 'Poison' : me.cabin ? 'Deck' : 'Cabin' } as Record<string, string>)[t?.kind ?? 'helm'] : me.mode === 'climb' ? (Math.abs(me.z - nestZ(me.mast)) < 1.2 ? 'Step off' : 'Down') : me.mode === 'nest' ? 'Mast' : riding ? 'Right hook' : 'Leave';
  $('reelT').style.display = me.mode === 'harpoon' && e?.tether ? 'block' : 'none';
  $('cutT').style.display = cut ? 'block' : 'none';
  if (toastTimer > 0) { toastTimer -= dt; if (toastTimer <= 0) toastEl.classList.remove('show'); }
  if (playing) updateCrewPanel();
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
function fitFov(vfov: number, aspect: number) {
  const hmin = 78 * Math.PI / 180;
  return Math.min(Math.max(vfov, 2 * Math.atan(Math.tan(hmin / 2) / aspect)), 105 * Math.PI / 180);
}

// ---------- Render ----------
function common(p: Program) {
  return p.v3('uCamPos', cam.eye).v3('uSunDir', SUN_DIR).v3('uSunCol', atm.sun).v3('uAmbTop', atm.ambT).v3('uAmbBot', atm.ambB)
    .v3('uFogCol', atm.fog).f('uFogDen', atm.fogDen).f('uFlash', flash).v3('uFlashDir', flashDir).f('uTime', time);
}
let curVP: Mat4 = IDENT;
function draw(mesh: Mesh, model: Mat4, wet = 0) {
  pSolid.m4('uModel', model).f('uWet', wet);
  G.bindVertexArray(mesh.vao); G.drawArrays(G.TRIANGLES, 0, mesh.count);
}
function shipModel(e: ShipEnt) {
  const ax = shipAxes(e.ship.psi, e.ship.th, e.ship.ph), m = fromAxes(ax.X, ax.Y, ax.Z, [e.ship.x, e.ship.y, e.ship.z]), S = e.ship.scale;
  return S === 1 ? m : mul(m, SC(S, S, S));
}
function shipInv(e: ShipEnt): Mat4 {
  const { X, Y, Z } = shipAxes(e.ship.psi, e.ship.th, e.ship.ph), p: V3 = [e.ship.x, e.ship.y, e.ship.z];
  const d = (a: V3) => -(a[0] * p[0] + a[1] * p[1] + a[2] * p[2]);
  const k = 1 / e.ship.scale;
  return new Float32Array([X[0] * k, Y[0] * k, Z[0] * k, 0, X[1] * k, Y[1] * k, Z[1] * k, 0, X[2] * k, Y[2] * k, Z[2] * k, 0, d(X) * k, d(Y) * k, d(Z) * k, 1]);
}
function beamModel(a: V3, b: V3, w: number, h: number): Mat4 {
  const d = sub(b, a), len = Math.hypot(d[0], d[1], d[2]) || 1e-3;
  return mul(TR(a[0], a[1], a[2]), RZ(Math.atan2(d[1], d[0])), RYm(-Math.asin(clamp(d[2] / len, -1, 1))), SC(len, w, h));
}
function drawAvatar(p: Player, model: Mat4, inv = 1) {
  const meshes = avatarMeshes[p.color % avatarMeshes.length];
  if (p.mode === 'glide') {
    const base = mul(TR(p.x, p.y, p.z - 1), RZ(p.yaw), RYm(-p.pitch));
    for (const [part, m] of avatarPose(base, { phase: 0, walk: 0, wave: 0, hold: 0, time, look: 0, swing: 0, climb: 0, down: 0, sword: false, glide: 1 })) draw(meshes[part], m);
    return;
  }
  if (p.mode === 'ride') {
    const mon = rideMonster(RIDE_BASE - p.on), f = mon && rideFrame(mon);
    if (!f) return;
    const base = mul(TR(f.pos[0], f.pos[1], f.pos[2]), RZ(f.yaw));
    for (const [part, m] of avatarPose(base, { phase: 0, walk: 0, wave: 0, hold: 0, time, look: 0, swing: 0, climb: 0, down: 0, sword: false, ride: 1 })) draw(meshes[part], m);
    return;
  }
  const hold = p.mode === 'helm' || p.mode === 'harpoon' ? 1 : 0;
  const yaw = p.mode === 'helm' ? 0 : p.mode === 'harpoon' ? p.aimYaw : p.yaw;
  const px = p.mode === 'harpoon' ? HARPOON.x - Math.cos(p.aimYaw) * 1.9 : p.x, py = p.mode === 'harpoon' ? -Math.sin(p.aimYaw) * 1.9 : p.y;
  const pz = p.mode === 'harpoon' ? deckZ(px, py) : p.z;
  const base = mul(model, TR(px, py, pz), SC(inv, inv, inv), RZ(yaw));
  const anim = { phase: p.phase, walk: p.walk, wave: Math.min(p.wave, 1), hold, time, look: p.mode === 'walk' || p.mode === 'nest' ? p.pitch : 0, swing: p.swing, climb: p.mode === 'climb' ? 1 : 0, down: p.mode === 'down' ? 1 : 0, sword: (p.mode === 'walk' || p.mode === 'nest' || p.mode === 'plank') && !p.carry, carry: p.carry };
  for (const [part, m] of avatarPose(base, anim)) draw(meshes[part], m);
}
function project(p: V3, vp: Mat4): [number, number] | null {
  const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12], y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13], w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (w < 0.1) return null;
  return [(x / w * 0.5 + 0.5) * innerWidth, (1 - (y / w * 0.5 + 0.5)) * innerHeight];
}

function drawShip(e: ShipEnt) {
  const model = shipModel(e);
  const G_ = !!e.ghost;
  draw(G_ ? ghostShipMesh : shipMesh, model);
  for (const mt of MASTS) for (const y of mt.yards) draw(G_ ? ghostSailMesh : sailMesh, mul(model, TR(mt.x + 0.75, 0, y.z), SC(y.w, y.w, y.h * e.ship.sailSet)));
  CANNONS.forEach((c, i) => draw(G_ ? ghostCannonMesh : cannonMesh, mul(model, TR(c.x, c.y, c.z), RZ(c.side > 0 ? Math.PI / 2 : -Math.PI / 2), TR(-e.recoil[i] * 0.8, 0, 0))));
  const harpooner = (me.mode === 'harpoon' && me.on === e.id) ? me : [...others.values()].find((o) => o.mode === 'harpoon' && o.on === e.id);
  const hy = harpooner?.aimYaw ?? 0, hp = harpooner?.aimPitch ?? 0.05;
  const hm = mul(model, TR(HARPOON.x, 0, HARPOON.z), RZ(hy), RYm(-hp));
  draw(G_ ? fireLauncherMesh : harpoonMesh, hm);
  if (e.harpoonCd <= 0 && !e.tether) draw(G_ ? fireballMesh : boltMesh, G_ ? mul(hm, TR(2.4, 0, 0), SC(0.5, 0.5, 0.5)) : mul(hm, TR(0, 0, 0.35)));
  draw(wheelMesh, mul(model, TR(HELM.x, 0, HELM.z), RXm(e.ship.rudder * 3), SC(2.0, 2.0, 2.0)));
  MASTS.forEach((mt, i) => draw(foldedWings, mul(model, TR(mt.x - mt.r + 0.1, 0, climbTop(i) - 1.6))));
  for (const p of [me, ...others.values()]) {
    if (p.on !== e.id || (p === me && (!playing || (camView === 'fp' && me.mode !== 'down') || me.mode === 'harpoon'))) continue;
    if (p === me && me.mode === 'helm' && camView !== 'tp') continue;
    drawAvatar(p, model, 1 / e.ship.scale);
  }
  const ends = tetherEnds(e);
  if (ends) draw(ropeMesh, beamModel(ends.tip, ends.end, 0.07, 0.07));
  drawVoyage(e);
}

function render() {
  const aspect = canvas.width / canvas.height;
  cam.fov = fitFov(cam.fov, aspect);
  const view = lookAt(cam.eye, [cam.eye[0] + cam.fwd[0], cam.eye[1] + cam.fwd[1], cam.eye[2] + cam.fwd[2]], cam.up);
  const vp = multiply(perspective(cam.fov, aspect, cam.near, 9000), view);
  curVP = vp;
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.viewport(0, 0, canvas.width, canvas.height);
  G.clearColor(0, 0, 0, 1); G.clear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT);
  if (inCave()) { renderCave(vp); return; }

  G.disable(G.DEPTH_TEST); G.depthMask(false);
  common(pSky.use()).v3('uFwd', cam.fwd).v3('uRight', cam.right).v3('uUp', cam.up).f('uTanHalf', Math.tan(cam.fov / 2)).f('uAspect', aspect)
    .v3('uSkyTop', atm.sky).v3('uIsland', [IC.x, IC.y, IC.R]).v2('uWindDir', windVec[0], windVec[1]).f('uStorm', smoothstep(0.1, 0.8, atm.s));
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  G.enable(G.DEPTH_TEST); G.depthMask(true);

  common(pSolid.use()).m4('uVP', vp);
  draw(land, IDENT, 1);
  if (Math.hypot(cam.eye[0] - EYE.x, cam.eye[1] - EYE.y) < 4000) draw(eyeIsland, IDENT, 1);
  BUOYS.forEach((b, i) => draw(buoyMeshes[b.kind], buoyModels[i]));
  const visible = [...ships.values()].filter((e) => !(e.preview && playing) && Math.hypot(e.ship.x - cam.eye[0], e.ship.y - cam.eye[1]) < 3000);
  for (const e of visible) drawShip(e);
  // Gliders and Leviathan riders live in world space.
  for (const p of [me, ...others.values()]) {
    if (p.on >= 0 || (p.mode !== 'glide' && p.mode !== 'ride' && p.mode !== 'land')) continue;
    if (p === me && (!playing || camView === 'fp')) continue;
    drawAvatar(p, IDENT);
  }
  drawVisionScene();
  for (const b of boards) {
    const ea = ships.get(b.a), eb = ships.get(b.b);
    if (!ea || !eb) continue;
    draw(plankMesh, beamModel(ea.ship.local(b.pa[0], b.pa[1], MAIN_DECK + 1.15), eb.ship.local(b.pb[0], b.pb[1], MAIN_DECK + 1.15), 1, 1));
  }
  for (const m of monsters.list) {
    for (let i = m.segs.length - 1; i >= 1; i--) {
      const p = m.segs[i], q = m.segs[i - 1], r = segRadius(i, m.scale);
      const yaw = Math.atan2(q[1] - p[1], q[0] - p[0]), pitch = Math.atan2(q[2] - p[2], Math.hypot(q[0] - p[0], q[1] - p[1]));
      draw(segMesh, mul(TR(p[0], p[1], p[2]), RZ(yaw), RYm(-pitch), SC(r * 1.25, r, r)));
    }
    if (m.segs.length > 1) {
      const h = m.segs[0], n = m.segs[1];
      const yaw = Math.atan2(h[1] - n[1], h[0] - n[0]), pitch = Math.atan2(h[2] - n[2], Math.hypot(h[0] - n[0], h[1] - n[1]));
      draw(headMesh, mul(TR(h[0], h[1], h[2]), RZ(yaw), RYm(-pitch * 0.7 - m.rear * 0.25), SC(m.scale, m.scale, m.scale)));
    }
  }
  for (const s of shots) {
    if (s.kind === 'f') draw(fireballMesh, TR(s.p[0], s.p[1], s.p[2]));
    else if (s.kind === 'c') draw(ballMesh, ships.get(s.sid)?.ghost ? mul(TR(s.p[0], s.p[1], s.p[2]), SC(1.5, 1.5, 1.5)) : TR(s.p[0], s.p[1], s.p[2]));
    else draw(boltMesh, mul(TR(s.p[0], s.p[1], s.p[2]), RZ(Math.atan2(s.v[1], s.v[0])), RYm(-Math.atan2(s.v[2], Math.hypot(s.v[0], s.v[1])))));
  }

  // Ocean, with water masked out inside every nearby hull.
  const near = visible.slice().sort((a, b) => Math.hypot(a.ship.x - cam.eye[0], a.ship.y - cam.eye[1]) - Math.hypot(b.ship.x - cam.eye[0], b.ship.y - cam.eye[1])).slice(0, 6);
  const inv = new Float32Array(16 * 6);
  near.forEach((e, i) => inv.set(shipInv(e), i * 16));
  const vk = visionK();
  if (vk > 0) updatePhases(time + vision.warp);
  common(pOcean.use()).f('uAmpBoost', vk > 0 ? mix(1, VISIONS[vision.kind].amp, vk) : 1).m4('uVP', vp).v2('uCenter', Math.round(cam.eye[0] / 4) * 4, Math.round(cam.eye[1] / 4) * 4)
    .v4a('uWA', uWA).v4a('uWB', uWB).v3('uIsland', [IC.x, IC.y, IC.R]).i('uWake', 0).i('uShore', 1)
    .v3('uWakeRect', [WAKE.ox, WAKE.oy, WAKE.size]).v3('uShoreRect', [IC.x + SHORE.min, IC.y + SHORE.min, SHORE.size])
    .v3('uSkyTop', atm.sky).v2('uWind', windVec[0] * 0.15, windVec[1] * 0.15).i('uShipN', near.length);
  const invLoc = pOcean.u.get('uShipInv');
  if (invLoc) G.uniformMatrix4fv(invLoc, false, inv);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  G.activeTexture(G.TEXTURE1); G.bindTexture(G.TEXTURE_2D, shoreTex);
  G.bindVertexArray(ocean.vao); G.drawElements(G.TRIANGLES, ocean.count, G.UNSIGNED_INT, 0);
  if (vk > 0) updatePhases(time);

  G.enable(G.BLEND); G.depthMask(false);
  G.blendFunc(G.SRC_ALPHA, G.ONE_MINUS_SRC_ALPHA);
  common(pSpray.use()).m4('uVP', vp).f('uPx', canvas.height / Math.tan(cam.fov / 2));
  G.bindVertexArray(sprayBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, sprayBuf.buf); G.bufferData(G.ARRAY_BUFFER, sprayPos, G.DYNAMIC_DRAW);
  G.drawArrays(G.POINTS, 0, MAXP);
  const rainCount = Math.floor((isMobile ? 2500 : 6000) * atm.rain);
  if (rainCount > 0) {
    const f = focusShip();
    common(pRain.use()).m4('uVP', vp).v3('uCamVel', [f.vx, f.vy, 0])
      .v3('uRainVel', [windVec[0] * 10 * atm.wind, windVec[1] * 10 * atm.wind, -11]).f('uBox', 44).m4('uBoatInv', IDENT).f('uMask', 0).f('uAmt', 0.4 + 0.6 * atm.rain);
    G.bindVertexArray(rainVao); G.drawArrays(G.LINES, 0, rainCount * 2);
  }

  G.blendFunc(G.ONE, G.ONE);
  const beamAng = time * 0.55;
  const bm = fromAxes([Math.cos(beamAng), Math.sin(beamAng), 0], [-Math.sin(beamAng), Math.cos(beamAng), 0], [0, 0, 1], [LIGHTHOUSE.x, LIGHTHOUSE.y, LIGHTHOUSE.lamp]);
  pBeam.use().m4('uVP', vp).m4('uModel', bm).v3('uCamPos', cam.eye).f('uFogDen', atm.fogDen).f('uI', 0.6 + 0.6 * (1 - atm.s));
  G.bindVertexArray(beam.vao); G.drawArrays(G.TRIANGLES, 0, beam.count);

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
  for (const e of visible) for (const y of [-2.2, 2.2]) { const p = e.ship.local(-18.4, y, 7.1); const d = dist3(p, cam.eye); addGlow(p, 1.2 + d * 0.004, e.ghost ? [0.35, 1, 0.6] : [1, 0.75, 0.4], (e.ghost ? 1.4 : 0.9) * Math.exp(-d * atm.fogDen * 0.5)); }
  for (const m of monsters.list) if (m.segs[0] && m.st !== MS.Dying) {
    const h = m.segs[0], hd = Math.atan2(h[1] - (m.segs[1]?.[1] ?? h[1]), h[0] - (m.segs[1]?.[0] ?? h[0])), s = m.scale;
    for (const k of [1, -1]) addGlow([h[0] + (Math.cos(hd) * 2.3 - Math.sin(hd) * k * 0.95) * s, h[1] + (Math.sin(hd) * 2.3 + Math.cos(hd) * k * 0.95) * s, h[2] + 0.6 * s], 0.9 * s, [1, 0.85, 0.2], 0.9 + m.hurt);
  }
  drawFires();
  visionGlows(); spiceGlows();
  for (const s of shots) if (s.kind === 'f') { const d = dist3(s.p, cam.eye); addGlow(s.p, 3.5 + d * 0.004, [1, 0.55, 0.15], 1.6); }
  if (glowCount) {
    pGlow.use().m4('uVP', vp).v3('uCamRight', cam.right).v3('uCamUp', cam.up);
    G.bindVertexArray(glowBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, glowBuf.buf);
    G.bufferData(G.ARRAY_BUFFER, glowData.subarray(0, glowCount * 60), G.DYNAMIC_DRAW);
    G.drawArrays(G.TRIANGLES, 0, glowCount * 6);
  }
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

function updateTags() {
  for (const o of others.values()) {
    if (!o.tag) { o.tag = document.createElement('div'); o.tag.className = 'tag'; tagsEl.appendChild(o.tag); }
    const label = `${o.name}${o.mode === 'down' ? ' (down)' : ''}`;
    if (o.tag.textContent !== label) o.tag.textContent = label;
    o.tag.style.color = o.crew === me.crew ? COAT_CSS[o.color % COAT_CSS.length] : '#e0624a';
    const w = playerWorld(o), head: V3 | null = w ? [w[0], w[1], w[2] + (o.mode === 'glide' ? 1.4 : 2.35)] : null;
    const s = head ? project(head, curVP) : null;
    const d = head ? dist3(head, cam.eye) : 1e9;
    if (s && d < 120 && playing) { o.tag.style.display = 'block'; o.tag.style.left = `${s[0]}px`; o.tag.style.top = `${s[1]}px`; o.tag.style.opacity = String(clamp(1.4 - d / 80, 0.3, 1)); }
    else o.tag.style.display = 'none';
  }
}

// ---------- Networking ----------
function remoteShipUpdate(e: ShipEnt, dt: number) {
  if (!e.snap) return;
  const a = e.snap.sh, age = clamp(time - e.snap.tm, 0, 0.35);
  const t = a.slice();
  t[0] += a[3] * age; t[1] += a[4] * age; t[2] += a[5] * age; t[6] += a[7] * age; t[8] += a[9] * age; t[10] += a[11] * age;
  if (!e.inited) { e.ship.unpack(t); e.inited = true; resetMarks(e); return; }
  const cur = e.ship.pack(), k = 1 - Math.exp(-dt * 12);
  const out = t.map((v, i) => ([0, 1, 6, 8, 10].includes(i) ? cur[i] + (v - cur[i]) * k : i === 2 ? angLerp(cur[i], v, k) : v));
  if (Math.hypot(t[0] - cur[0], t[1] - cur[1]) > 30) { e.ship.unpack(t); resetMarks(e); return; }
  e.ship.unpack(out);
}
function sendShip(e: ShipEnt) {
  if (e.cave) return; // still down in the lair: nobody else knows of her yet
  const t = e.tether;
  net.send({
    t: 'S', sid: e.id, own: e.owner, nm: e.name, cap: e.captain, tm: +time.toFixed(3), sh: e.ship.pack().map((v) => +v.toFixed(4)), hm: e.helm, sc: e.score,
    th: t ? [t.kind === 'm' ? 1 : 2, t.target, +t.len.toFixed(1), +t.ap[0].toFixed(2), +t.ap[1].toFixed(2), +t.ap[2].toFixed(2), +t.t0.toFixed(2)] : null,
    hc: +e.harpoonCd.toFixed(2), bd: boards.filter((b) => b.a === e.id), gh: e.ghost ? 1 : 0, vo: packVoyage(e),
  });
}
function reassignShips() {
  const present = new Set([myId(), ...net.peers]);
  for (const e of [...ships.values()]) {
    if (e.preview || present.has(e.owner)) continue;
    const crew = [me, ...others.values()].filter((p) => p.crew === e.id && present.has(p.id)).map((p) => p.id).sort((a, b) => a - b);
    if (!crew.length) { ships.delete(e.id); boards = boards.filter((b) => b.a !== e.id && b.b !== e.id); continue; }
    e.owner = crew[0];
    if (e.owner === myId()) { e.snap = null; if (crew.length > 1 || me.crew === e.id) toast(`You now command the ${e.name}`, 3); }
  }
}
net.onChange = () => {
  for (const id of [...others.keys()]) if (!net.peers.has(id)) { const o = others.get(id)!; o.tag?.remove(); others.delete(id); for (const e of ships.values()) if (e.helm === id) e.helm = null; if (o.crew === me.crew) toast(`${o.name} left the ship`, 2.5); }
  reassignShips();
  crewPanelKey = '';
};
net.onMsg = (m: NetMsg) => {
  const from = m.from as number;
  switch (m.t) {
    case 'p': {
      let o = others.get(from);
      if (!o) { o = newPlayer(from, String(m.n), Number(m.c)); others.set(from, o); if (playing && m.cr === me.crew) toast(`${o.name} came aboard`, 2.5); }
      o.name = String(m.n); o.color = Number(m.c); o.crew = m.cr as number; o.hp = m.hp as number; o.seen = now();
      const prevMode = o.mode;
      o.mode = m.md as Mode; o.moving = !!m.mv; o.reel = !!m.rl; o.mast = (m.ms as number) ?? 0; o.hookMask = (m.hk as number) ?? 0; o.cabin = !!m.cb; o.carry = !!m.po;
      if (o.snaps.length && o.snaps[o.snaps.length - 1].on !== m.on) o.snaps = [];
      o.snaps.push({ t: now(), x: m.x as number, y: m.y as number, z: m.z as number, yaw: m.yw as number, pitch: m.pt as number, ay: m.ay as number, ap: m.ap as number, on: m.on as number });
      if (o.snaps.length > 8) o.snaps.shift();
      if (m.wv) o.wave = 1.6;
      if (m.sw && o.swing === 0) { o.swing = 0.01; const w = playerWorld(o); audio.whoosh(w ? dist3(w, cam.eye) : 50); }
      if (prevMode !== 'down' && o.mode === 'down') { const w = playerWorld(o); if (w && dist3(w, cam.eye) < 60) toast(`${o.name} is down!`, 2); }
      break;
    }
    case 'S': {
      const sid = m.sid as number;
      let e = ships.get(sid);
      if (!e) { e = newShipEnt(sid, m.own as number, String(m.nm), String(m.cap ?? '')); ships.set(sid, e); }
      if (m.gh && !e.ghost) makeGhost(e);
      if (owns(e) && m.own !== myId()) e.owner = m.own as number;
      if (owns(e)) break;
      unpackVoyage(e, (m.vo as Parameters<typeof unpackVoyage>[1]) ?? null);
      e.owner = m.own as number; e.name = String(m.nm); e.captain = String(m.cap ?? ''); e.seen = now();
      e.snap = { tm: m.tm as number, sh: m.sh as number[] };
      const prevHelm = e.helm;
      e.helm = (m.hm as number | null) ?? null; e.score = m.sc as number; e.harpoonCd = m.hc as number;
      const th = m.th as number[] | null;
      e.tether = th ? { kind: th[0] === 1 ? 'm' : 's', target: th[1], len: th[2], ap: [th[3], th[4], th[5]], t0: th[6] } : null;
      if (e.id === me.on) {
        if (e.helm === me.id && me.mode !== 'helm') enterHelm();
        if (prevHelm === me.id && e.helm !== me.id && me.mode === 'helm') { me.mode = 'walk'; me.x = HELM.stand[0] - 0.8; }
      }
      for (const b of (m.bd as Board[]) ?? []) if (!wasCut(b.a, b.b) && !boards.some((x) => x.a === b.a && x.b === b.b)) boards.push(b);
      break;
    }
    case 'M': {
      if (net.role === 'guest') {
        const off = (m.tm as number) - now();
        clockOffset = clockOffset === null ? off : clockOffset + (off - clockOffset) * 0.05;
        monsterRows = m.mo as number[][]; monsterTm = m.tm as number;
      }
      break;
    }
    case 'claim': { const e = ships.get(m.sid as number); if (e && owns(e) && e.helm === null) e.helm = from; break; }
    case 'release': { const e = ships.get(m.sid as number); if (e && owns(e) && e.helm === from) e.helm = null; break; }
    case 'in': { const e = ships.get(m.sid as number); if (e && owns(e) && e.helm === from) e.remote = { rudder: m.r as number, sail: m.s as number, back: !!m.b }; break; }
    case 'fire': { const e = ships.get(m.sid as number); if (e && owns(e)) hostFire(e, m.k as ShotKind, m.i as number, m.ay as number, m.ap as number); break; }
    case 'unhook': { const e = ships.get(m.sid as number); if (e && owns(e)) releaseTether(e, true); break; }
    case 'shot': if (!shots.some((s) => s.id === m.id)) spawnShot(m.id as number, m.sid as number, m.k as ShotKind, m.i as number, m.p as V3, m.v as V3); break;
    case 'mhit': onMonsterHit(m as unknown as HitMsg); break;
    case 'shit': onShipHit(m as unknown as ShipHitMsg); break;
    case 'rip': onRip(m as unknown as RipMsg); break;
    case 'ram': onRamMsg(m.sid as number, m.p as V3, !!m.boss); break;
    case 'score': { const e = ships.get(m.sid as number); if (e && owns(e)) e.score += m.add as number; break; }
    case 'board': cutAt.delete(pairKey(m.a as number, m.b as number)); if (!boards.some((b) => b.a === m.a && b.b === m.b)) { boards.push({ a: m.a as number, b: m.b as number, pa: m.pa as [number, number], pb: m.pb as [number, number] }); if ([m.a, m.b].includes(me.on)) toast('Grappled! Prepare to be boarded', 2.5); } break;
    case 'unboard': onUnboard(m.a as number, m.b as number); break;
    case 'slash': { audio.cut(0); takeDamage(m.dmg as number, `${others.get(from)?.name ?? 'A pirate'} cut you down`); break; }
    case 'ev': { const e = ships.get(m.sid as number); if (e && !owns(e)) e.ship.events.push({ type: m.k as ShipEvent['type'], i: m.i as number }); break; }
    case 'steer': if (isWorldHost()) monsters.steer(m.m as number, from, m.rs as number, m.tu as number, m.ga as number); break;
    case 'sound': if (isWorldHost()) monsters.sound(m.m as number); break;
    case 'oslam': onSlam(m as unknown as { sid: number; i: number; x: number; y: number }); break;
    case 'opois': onPoison(m as unknown as { sid: number; i: number; hp: number }); break;
    case 'poison': { const e = ships.get(m.sid as number); if (e && owns(e)) poisonOcto(e, m.i as number); break; }
    case 'skhit': { const e = ships.get(m.sid as number); if (e && owns(e)) hitSkeleton(e, m.k as number); break; }
    case 'skslash': takeDamage(m.dmg as number, 'A skeleton cut you down'); audio.cut(0); break;
    case 'own': { const e = ships.get(m.sid as number); if (e) { e.owner = m.to as number; e.helm = null; if (e.owner === myId()) { e.snap = null; toast(`You now command the ${e.name}`, 3); } } break; }
    case 'gone': { const sid = m.sid as number; ships.delete(sid); boards = boards.filter((b) => b.a !== sid && b.b !== sid); if (me.on === sid && me.crew !== sid) { me.on = me.crew; me.mode = 'walk'; me.x = -6; me.y = 0; me.z = MAIN_DECK; } break; }
    case 'wreck': if ((m.sid as number) === me.on) showWreck(); break;
    case 'boss': toast('The Leviathan rises!', 4); break;
  }
};
let monsterRows: number[][] | null = null, monsterTm = 0;
function interpOthers() {
  const t = now() - 0.12;
  for (const o of others.values()) {
    const s = o.snaps;
    if (!s.length) continue;
    let a = s[0], b = s[s.length - 1];
    for (let i = 0; i < s.length - 1; i++) if (s[i].t <= t && s[i + 1].t >= t) { a = s[i]; b = s[i + 1]; break; }
    const f = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1) : 1;
    const nx = a.x + (b.x - a.x) * f, ny = a.y + (b.y - a.y) * f;
    const moved = Math.hypot(nx - o.x, ny - o.y);
    o.on = b.on; o.x = nx; o.y = ny; o.z = a.z + (b.z - a.z) * f; o.yaw = angLerp(a.yaw, b.yaw, f); o.pitch = a.pitch + (b.pitch - a.pitch) * f;
    o.aimYaw = a.ay + (b.ay - a.ay) * f; o.aimPitch = a.ap + (b.ap - a.ap) * f;
    o.walk += ((o.moving ? 1 : 0) - o.walk) * 0.2;
    o.phase += Math.min(moved, 1) * 2.4 + (o.mode === 'climb' && o.moving ? 0.1 : 0);
    if (o.swing > 0) { o.swing += 1 / 27; if (o.swing >= 1) o.swing = 0; }
    o.wave = Math.max(0, o.wave - 1 / 60);
    if (o.moving && o.mode === 'walk' && Math.floor(o.phase / Math.PI) !== o.lastStep) { o.lastStep = Math.floor(o.phase / Math.PI); const w = playerWorld(o); if (w) audio.step(dist3(w, cam.eye)); }
  }
}
let sendT = 0, shipT = 0, monT = 0;
function netSend(dt: number) {
  if (!net.online) return;
  sendT += dt; shipT += dt; monT += dt;
  if (playing && sendT > 1 / 15) {
    sendT = 0;
    net.send({ t: 'p', n: me.name, c: me.color, cr: me.crew, on: me.on, x: +me.x.toFixed(2), y: +me.y.toFixed(2), z: +me.z.toFixed(2), yw: +me.yaw.toFixed(3), pt: +me.pitch.toFixed(3), md: me.mode, mv: me.moving ? 1 : 0, wv: me.wave > 1.5 ? 1 : 0, sw: me.swing > 0 && me.swing < 0.2 ? 1 : 0, ay: +me.aimYaw.toFixed(3), ap: +me.aimPitch.toFixed(3), hp: Math.round(me.hp), rl: me.reel ? 1 : 0, ms: me.mast, hk: me.hookMask, cb: me.cabin ? 1 : 0, po: me.carry ? 1 : 0 });
    const e = onShip();
    if (e && !owns(e) && me.mode === 'helm') net.to(e.owner, { t: 'in', sid: e.id, r: +helmRudder.toFixed(3), s: +helmSail.toFixed(3), b: helmBack ? 1 : 0 });
  }
  if (shipT > 1 / 20) { shipT = 0; for (const e of ships.values()) if (owns(e) && !e.preview) sendShip(e); }
  if (isWorldHost() && monT > 1 / 10) { monT = 0; net.send({ t: 'M', tm: +time.toFixed(3), mo: monsters.pack() }); }
}

// ---------- Wreck / rams ----------
function showWreck() { document.body.classList.add('wrecked'); setTimeout(() => document.body.classList.remove('wrecked'), 4500); }
function onRamMsg(sid: number, p: V3, boss: boolean) {
  const e = ships.get(sid);
  if (e && owns(e)) {
    const { Y } = shipAxes(e.ship.psi, e.ship.th, e.ship.ph);
    const l = toLocal(e, p), side = Math.sign(l[1]) || 1, k = boss ? 3 : 1;
    e.ship.impulse(-Y[0] * side * 1.6 * k, -Y[1] * side * 1.6 * k, -side * 0.18 * k);
    e.ship.hull = Math.max(0, e.ship.hull - (boss ? 28 : 12));
  }
  const d = dist3(p, cam.eye);
  audio.crunch(1 / (1 + d / 60)); audio.roar(d);
  if (me.on === sid) { cam.shake = Math.max(cam.shake, boss ? 0.6 : 0.35); toast(boss ? 'The Leviathan strikes!' : 'Rammed!', 1.8); if (boss && (me.mode === 'climb' || me.mode === 'nest')) { me.mode = 'walk'; me.x = climbX(me.mast) - 0.6; me.y = 0; me.z = deckZ(me.x, 0); takeDamage(25, 'Thrown from the rigging!'); } }
  burst(p, boss ? 220 : 100, boss ? 20 : 12, boss ? 22 : 14, 1.4);
}

// ---------- Game step ----------
function handleShipEvents(e: ShipEnt) {
  // The owner's simulation produces the events; share them so the whole crew feels them.
  if (owns(e) && !e.preview && net.online) for (const ev of e.ship.events) net.send({ t: 'ev', sid: e.id, k: ev.type, i: +ev.i.toFixed(2) });
  for (const ev of e.ship.events) {
    const d = dist3([e.ship.x, e.ship.y, e.ship.z], cam.eye);
    if (ev.type === 'slam') {
      if (d < 400) audio.slap(ev.i / (1 + d / 80));
      if (me.on === e.id) { cam.shake = Math.max(cam.shake, 0.06 * ev.i); wakeBlob = 0.5 * ev.i; }
      if (d < 600) for (let k = 0; k < 40 + 120 * ev.i; k++) emitLocal(e, [15 + R() * 3, (R() - 0.5) * 6, 3 + R()], [-(1 + R() * 6 * ev.i), (R() - 0.5) * (6 + 10 * ev.i), 4 + R() * (6 + 12 * ev.i)], 1.2);
    } else if (ev.type === 'splash') {
      splashDown(e, ev.i, d);
    } else if (ev.type === 'green') {
      if (d < 600) for (let k = 0; k < 120 * ev.i; k++) emitLocal(e, [8 + R() * 8, (R() - 0.5) * 7, 5 + R() * 2], [-(5 + R() * 6), (R() - 0.5) * 4, 2 + R() * 4], 1.2);
    } else if (ev.type === 'ground') {
      // Grinding up the beach: sand and spray thrown off the forefoot, the hull shuddering.
      if (d < 400) { audio.crunch(0.35 * ev.i / (1 + d / 80)); audio.slap(0.4 * ev.i / (1 + d / 80)); }
      if (d < 600) for (let k = 0; k < 50 * ev.i; k++) emitLocal(e, [10 + R() * 8, (R() - 0.5) * 8, 0.5], [(R() - 0.5) * 6, (R() - 0.5) * 10, 3 + R() * 6], 1.1);
      if (me.on === e.id) {
        cam.shake = Math.max(cam.shake, 0.25 * ev.i);
        if (time - (e.groundT ?? -99) > 30) toast('Run aground on the island!', 3);
      }
      e.groundT = time;
    } else { audio.crunch(ev.i / (1 + d / 60)); if (me.on === e.id) { cam.shake = Math.max(cam.shake, 0.3 * ev.i); toast('Collision!', 1.5); } }
  }
  e.ship.events.length = 0;
}

// ---------- Splash-down ----------
const splash = { t: 99, i: 0, on: -1 };
const stagger = { x: 0, y: 0 };
const drenchEl = $('drench');
function splashDown(e: ShipEnt, i: number, d: number) {
  if (d < 500) audio.splashdown(i / (1 + d / 120));
  if (d < 700) {
    // White sheets thrown out from both sides of the hull, a wall at the bow and water sluicing aft.
    for (const sd of [1, -1]) for (let k = 0; k < 220 * i; k++) {
      const x = -6 + R() * 23;
      emitLocal(e, [x, sd * (halfBeam(x) + 0.3), 0.6 + R()], [-(1 + R() * 4), sd * (5 + R() * 11 * i), 6 + R() * 14 * i], 1.6);
    }
    for (let k = 0; k < 160 * i; k++) emitLocal(e, [16 + R() * 3, (R() - 0.5) * 5, 1.5 + R() * 2], [2 + R() * 4, (R() - 0.5) * 12 * i, 12 + R() * 14 * i], 1.8);
    for (let k = 0; k < 140 * i; k++) emitLocal(e, [7 + R() * 9, (R() - 0.5) * 7, MAIN_DECK + 0.6 + R() * 1.5], [-(6 + R() * 6), (R() - 0.5) * 3, 1 + R() * 3], 1.2);
  }
  if (me.on !== e.id) return;
  wakeBlob = Math.max(wakeBlob, i);
  cam.shake = Math.max(cam.shake, 0.35 * i);
  splash.t = 0; splash.i = i; splash.on = e.id;
  // Crew are thrown forward; anyone aloft loses their footing.
  stagger.x = 3.5 * i; stagger.y = (R() - 0.5) * 2 * i;
  if (me.mode === 'climb' && i > 0.6) me.z = Math.max(MASTS[me.mast].base, me.z - 2.5 * i);
  if (i > 0.75 && (me.mode === 'walk' || me.mode === 'nest')) toast('Brace!', 1.2);
}
function updateSplashCam(dt: number) {
  splash.t += dt;
  stagger.x *= Math.exp(-dt * 3); stagger.y *= Math.exp(-dt * 3);
  const t = splash.t, i = splash.i, live = playing && splash.on === me.on && t < 6;
  if (live) {
    // The deck drops away under you and bounces back.
    const dip = -i * 0.9 * Math.exp(-t * 3.2) * Math.cos(t * 7);
    cam.eye = [cam.eye[0] + cam.up[0] * dip, cam.eye[1] + cam.up[1] * dip, cam.eye[2] + cam.up[2] * dip];
    cam.fov += i * 0.08 * Math.exp(-t * 4);
  }
  const close = live && !(me.mode === 'helm' && camView === 'tp');
  const w = close ? i * Math.min(1, t * 6) * Math.exp(-t * 0.7) : 0;
  drenchEl.style.opacity = w.toFixed(3);
  drenchEl.style.display = w > 0.01 ? 'block' : 'none';
}

function stepOwnedShip(e: ShipEnt, dt: number) {
  if (e.helm === myId() && me.on === e.id && me.mode === 'helm') { e.ship.rudder = helmRudder; e.ship.sail = helmSail; e.ship.backwater = helmBack; }
  else if (e.helm !== null) { e.ship.rudder += clamp(e.remote.rudder - e.ship.rudder, -3 * dt, 3 * dt); e.ship.sail = clamp(e.remote.sail, 0, 1); e.ship.backwater = !!e.remote.back; }
  else { e.ship.backwater = false; e.ship.rudder *= Math.exp(-dt * 0.8); }
  if (e.helm !== null && e.helm !== myId() && !others.has(e.helm)) e.helm = null;
  if (e.helm !== null && e.helm !== myId()) { const h = others.get(e.helm); if (h && (h.mode === 'down' || h.on !== e.id)) e.helm = null; }
  if (e.helm === myId() && (me.mode !== 'helm' || me.on !== e.id)) e.helm = null;
  const n = Math.max(1, Math.ceil(dt / (1 / 60)));
  for (let i = 0; i < n; i++) e.ship.step(dt / n, windVec[0] * 14 * atm.wind, windVec[1] * 14 * atm.wind);
  if (e.ship.hull <= 0 && e.wreckT <= 0) {
    e.wreckT = 4.5; net.send({ t: 'wreck', sid: e.id });
    if (me.on === e.id) showWreck();
    for (const b of boards.filter((x) => x.a === e.id || x.b === e.id)) { net.send({ t: 'unboard', a: b.a, b: b.b }); onUnboard(b.a, b.b); }
    e.tether = null;
  }
  if (e.wreckT > 0) { e.wreckT -= dt; if (e.wreckT <= 0) { placeShip(e, 'shelter', true); e.score = 0; } }
}

function simulate(dt: number) {
  if (net.role === 'guest' && clockOffset !== null) time = now() + clockOffset; else time += dt;
  updatePhases(time);
  if (playing && me.mode !== 'down') updateControls(dt);
  if (me.mode !== 'down' && me.hp < 100 && !inBoarding()) me.hp = Math.min(100, me.hp + 1.5 * dt); // slow recovery
  if (me.mode === 'down') { downT += dt; if (downT > 8 && !inBoarding()) revive(); }
  updatePlank(dt);
  for (const e of ships.values()) { if (e.cave) updateCaveShip(e, dt); else if (owns(e) || e.preview) stepOwnedShip(e, dt); else remoteShipUpdate(e, dt); handleShipEvents(e); }
  updateDeep(dt);
  for (const e of ships.values()) { if (owns(e) && !e.preview && !e.cave) updateVoyage(e, dt); noticeVoyage(e); }
  updateSlams(dt);
  updateTethers(dt);
  updateBoardsAndCollisions(dt);
  if (isWorldHost()) {
    const refs = [...ships.values()].filter((e) => !e.preview).map((e) => ({ id: e.id, x: e.ship.x, y: e.ship.y, psi: e.ship.psi, vx: e.ship.vx, vy: e.ship.vy }));
    monsters.update(dt, refs, stormAt, time, {
      ram: (m, sid) => { const p = (m.segs[0] ?? [m.x, m.y, 0]).map((x) => +x.toFixed(1)) as V3; net.send({ t: 'ram', sid, p, boss: m.boss ? 1 : 0 }); onRamMsg(sid, p, m.boss); },
      bossSpawn: () => { net.send({ t: 'boss' }); toast('The Leviathan rises!', 4); },
    });
  } else if (monsterRows) monsters.apply(monsterRows, dt, time, monsterTm);
  trackMonsterDeaths();
  for (const e of ships.values()) {
    if (Math.hypot(e.ship.x - cam.eye[0], e.ship.y - cam.eye[1]) > 400) continue;
    const speed = Math.hypot(e.ship.vx, e.ship.vy);
    if (speed > 3 && e.ship.imm > -3) for (const sg of [1, -1]) if (R() < 0.6) emitLocal(e, [14 + R() * 2, sg * 3.2, 0.8], [-1 - R(), sg * (2 + R() * 3), 1.5 + R() * 3], 0.8);
  }
  updateShots(dt);
  updateFires(dt);
  interpOthers();
  updateAtmosphere(dt);
  updateLightning(dt);
  updateBuoys();
  updateSpray(dt);
  updateWake();
  updateCamera(dt);
  updateSplashCam(dt);
  updateVision(dt);
  visionCamera();
  if (playing) checkGates();
  // Boss music when the Leviathan is near my ship.
  const f = focusShip();
  const boss = monsters.list.find((m) => m.boss && m.st !== MS.Dying);
  audio.setBoss(boss && playing ? clamp(1.6 - Math.hypot(boss.x - f.x, boss.y - f.y) / 900, 0, 1) : 0);
  // Drop ships we no longer hear from.
  for (const e of [...ships.values()]) if (!owns(e) && !e.preview && now() - e.seen > 6) { ships.delete(e.id); boards = boards.filter((b) => b.a !== e.id && b.b !== e.id); }
}
const deadSeen = new Set<number>();
function trackMonsterDeaths() {
  for (const m of monsters.list) if (m.st === MS.Dying && !deadSeen.has(m.id)) {
    deadSeen.add(m.id);
    const d = m.segs[0] ? dist3(m.segs[0], cam.eye) : 1e9;
    if (d < 1500) { toast(m.boss ? 'The Leviathan is slain!' : 'Serpent slain!', m.boss ? 5 : 2.5); audio.roar(d, true); if (m.segs[0]) burst(m.segs[0], m.boss ? 300 : 120, m.boss ? 22 : 14, m.boss ? 24 : 16, 1.6); }
  }
}

let last = performance.now();
function frame(nowMs: number) {
  const dt = Math.min((nowMs - last) / 1000, 0.05);
  last = nowMs;
  resize();
  simulate(dt);
  render();
  updateTags();
  updateHud(dt);
  netSend(dt);
  const f = focusShip();
  audio.update({ speed: me.mode === 'glide' ? Math.min(Math.hypot(me.gv[0], me.gv[1], me.gv[2]) / 25, 1) : Math.min(Math.hypot(f.vx, f.vy) / 8, 1), storm: 0.15 + 0.85 * atm.s, rain: atm.rain, wind: atm.wind, heave: Math.abs(f.vz), roll: Math.abs(f.phv) }, dt);
  if (!playing) updateLobby();
  requestAnimationFrame(frame);
}

// ---------- Lobby ----------
const PIRATE = ['Salty', 'One-Eyed', 'Red', 'Black', 'Mad', 'Calico', 'Iron', 'Grim', 'Bonny', 'Silver'];
const NAMES = ['Anne', 'Jack', 'Morgan', 'Ned', 'Mary', 'Flint', 'Rackham', 'Grace', 'Teach', 'Kidd'];
const nameEl = $<HTMLInputElement>('name');
let savedName = '';
try { savedName = localStorage.getItem('seas-name') ?? ''; } catch { /* storage blocked */ }
nameEl.value = savedName || `${PIRATE[Math.floor(R() * PIRATE.length)]} ${NAMES[Math.floor(R() * NAMES.length)]}`;
const inviteShip = Number(new URLSearchParams(location.search).get('ship') ?? 0);
let spawnMode: 'shelter' | 'storm' = 'shelter';
const setSpawn = (m: 'shelter' | 'storm') => { spawnMode = m; $('spShelter').classList.toggle('on', m === 'shelter'); $('spStorm').classList.toggle('on', m === 'storm'); };
$('spShelter').addEventListener('click', () => setSpawn('shelter'));
$('spStorm').addEventListener('click', () => setSpawn('storm'));
const statusEl = $('status'), crewsEl = $('crews');
let lobbyKey = '', lobbyT = 0;
function updateLobby() {
  lobbyT -= 1 / 60;
  if (lobbyT > 0) return;
  lobbyT = 0.5;
  const list = [...ships.values()].filter((e) => !e.preview);
  const rows = list.map((e) => {
    const crew = [...others.values()].filter((p) => p.crew === e.id);
    const s = stormAt(e.ship.x, e.ship.y);
    const where = Math.hypot(e.ship.x - BOSS_POS.x, e.ship.y - BOSS_POS.y) < 900 ? 'hunting the Leviathan' : s < 0.06 ? 'in the bay' : s < 0.5 ? 'in rough water' : 'in the storm';
    return { id: e.id, html: `<div class="crew${e.id === inviteShip ? ' invited' : ''}"><div><b>${esc(e.name)}</b> <span class="role">· ${crew.length || 1} aboard · ${where}</span><div class="role">${esc(crew.map((p) => p.name).join(', ') || e.captain)}</div></div><button class="go small" data-join="${e.id}">Join crew</button></div>` };
  });
  const key = JSON.stringify([rows.map((r) => r.html), net.online]);
  if (key === lobbyKey) return;
  lobbyKey = key;
  crewsEl.innerHTML = rows.length ? `<div class="lbl">Ships at sea</div>${rows.map((r) => r.html).join('')}` : net.online ? '<div class="role">No ships at sea yet. Be the first.</div>' : '';
  crewsEl.querySelectorAll<HTMLButtonElement>('[data-join]').forEach((b) => b.addEventListener('click', () => joinCrew(Number(b.dataset.join))));
}
function begin() {
  me.name = (nameEl.value.trim() || 'Pirate').slice(0, 18);
  try { localStorage.setItem('seas-name', me.name); } catch { /* storage blocked */ }
  me.id = myId();
  playing = true;
  document.body.classList.add('playing');
  audio.start();
  ships.delete(0);
  me.mode = 'walk'; me.hp = 100; me.yaw = 0; me.pitch = 0; me.z = MAIN_DECK;
  const f = focusShip(); gatePrev = [f.x, f.y]; WAKE.ox = Math.round(f.x); WAKE.oy = Math.round(f.y);
  crewPanelKey = '';
  if (!isTouch) lockPointer();
}
function sailOwn() {
  me.id = myId();
  const e = newShipEnt(me.id, me.id, shipName(me.id), (nameEl.value.trim() || 'Pirate').slice(0, 18));
  ships.set(e.id, e);
  placeShip(e, spawnMode, true);
  me.crew = me.on = e.id; me.color = 0; me.x = -7; me.y = -1.2;
  begin();
  if (WORLD_ROOM === 'WORLD1') history.replaceState(null, '', location.pathname);
  toast(spawnMode === 'storm' ? 'High seas' : 'Sheltered water', 3);
}
function joinCrew(sid: number) {
  const e = ships.get(sid);
  if (!e) { statusEl.textContent = 'That ship is no longer at sea.'; return; }
  me.id = myId();
  me.crew = me.on = sid; me.color = 1 + (me.id % 3); me.x = -5 - Math.random() * 3; me.y = (Math.random() - 0.5) * 3;
  begin();
  toast(`Welcome aboard the ${e.name}`, 3);
}
$('soloBtn').addEventListener('click', sailOwn);
$('camBtn').addEventListener('click', toggleCam);
$('muteBtn').addEventListener('click', toggleMute);
$('fsBtn').addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen?.().catch(() => undefined);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) last = performance.now(); });
// Hidden tabs get no animation frames: keep simulating and broadcasting from a timer so our ship stays in the world.
setInterval(() => {
  const t = performance.now();
  if (t - last < 400) return;
  let dt = Math.min((t - last) / 1000, 2);
  last = t;
  while (dt > 0) { const h = Math.min(dt, 0.05); simulate(h); netSend(h); dt -= h; }
}, 250);

// A local preview ship for the title screen (never broadcast).
const preview = newShipEnt(0, 0, 'preview', '');
preview.preview = true;
ships.set(0, preview);
placeShip(preview, 'shelter', false);
preview.owner = -1;
updateAtmosphere(10);
statusEl.textContent = 'Looking for ships at sea…';
net.connect(WORLD_ROOM).then(() => { statusEl.textContent = ''; crewPanelKey = ''; lobbyKey = ''; }).catch(() => { statusEl.textContent = 'Could not reach the other ships: sailing offline.'; });
requestAnimationFrame(frame);

// Dev hooks: deterministic stepping while the tab is throttled, and state inspection.
const dev = window as unknown as Record<string, unknown>;
dev.__advance = (n: number, dt = 1 / 60) => {
  for (let i = 0; i < n; i++) { simulate(dt); netSend(dt); }
  last = performance.now(); // keep the hidden-tab timer from also stepping
  render(); updateTags(); updateHud(0);
};
dev.__game = {
  get boards() { return boards; }, get vision() { return vision; }, startVision, stageAt, pourTarget, enterLair, sound, jump, fires, addFire, surge, ships, me, others, monsters, keys, cam, atm, net, shots, sailOwn, joinCrew, interact, fireAction, toggleCam, wave, slash, cutGangplank,
  setSpawn, look: (dx: number, dy: number) => { look.dx += dx; look.dy += dy; }, reel: (v: boolean) => { reelHeld = v; },
  ship: () => onShip(), place: (mode: 'shelter' | 'storm') => { const e = crewShip(); if (e) placeShip(e, mode, false); },
};
