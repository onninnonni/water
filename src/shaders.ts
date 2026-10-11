export const VERT = `#version 300 es
precision highp float;
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const HEADER = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 frag;
uniform vec2 uTexel;`;

// Semi-Lagrangian advection with dissipation.
export const ADVECT = HEADER + `
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform float uDt;
uniform float uDissipation;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 coord = vUv - uDt * vel * uTexel;
  coord = clamp(coord, uTexel * 0.5, 1.0 - uTexel * 0.5);
  frag = texture(uSource, coord) * uDissipation;
}`;

// Gaussian splat of velocity / dye.
export const SPLAT = HEADER + `
uniform sampler2D uTarget;
uniform vec2 uPoint;
uniform vec3 uValue;
uniform float uRadius;
uniform float uAspect;
void main() {
  vec2 d = vUv - uPoint;
  d.x *= uAspect;
  float g = exp(-dot(d, d) / uRadius);
  vec3 base = texture(uTarget, vUv).xyz;
  frag = vec4(base + uValue * g, 1.0);
}`;

export const DIVERGENCE = HEADER + `
uniform sampler2D uVelocity;
void main() {
  float L = texture(uVelocity, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uVelocity, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uVelocity, vUv - vec2(0.0, uTexel.y)).y;
  float T = texture(uVelocity, vUv + vec2(0.0, uTexel.y)).y;
  // Reflecting (solid) walls: normal velocity is zero at the boundary.
  if (vUv.x - uTexel.x < 0.0) L = 0.0;
  if (vUv.x + uTexel.x > 1.0) R = 0.0;
  if (vUv.y - uTexel.y < 0.0) B = 0.0;
  if (vUv.y + uTexel.y > 1.0) T = 0.0;
  frag = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}`;

// Generic Jacobi iteration: x = (xL + xR + xB + xT + alpha * b) * rBeta
export const JACOBI = HEADER + `
uniform sampler2D uX;
uniform sampler2D uB;
uniform float uAlpha;
uniform float uRBeta;
void main() {
  vec4 L = texture(uX, vUv - vec2(uTexel.x, 0.0));
  vec4 R = texture(uX, vUv + vec2(uTexel.x, 0.0));
  vec4 B = texture(uX, vUv - vec2(0.0, uTexel.y));
  vec4 T = texture(uX, vUv + vec2(0.0, uTexel.y));
  vec4 b = texture(uB, vUv);
  frag = (L + R + B + T + uAlpha * b) * uRBeta;
}`;

export const GRADIENT_SUBTRACT = HEADER + `
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  vec2 vel = texture(uVelocity, vUv).xy;
  vel -= 0.5 * vec2(R - L, T - B);
  frag = vec4(vel, 0.0, 1.0);
}`;

// Enforce reflecting walls: kill the normal component in the boundary cells.
export const BOUNDARY = HEADER + `
uniform sampler2D uVelocity;
void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  if (vUv.x < uTexel.x || vUv.x > 1.0 - uTexel.x) vel.x = -vel.x;
  if (vUv.y < uTexel.y || vUv.y > 1.0 - uTexel.y) vel.y = -vel.y;
  frag = vec4(vel, 0.0, 1.0);
}`;

// Water rendering: dye acts as a surface-height perturbation; shade with
// normals from its gradient, plus the pressure field for a wave-like look.
export const DISPLAY = HEADER + `
uniform sampler2D uDye;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform float uTime;
void main() {
  float hL = texture(uDye, vUv - vec2(uTexel.x, 0.0)).x;
  float hR = texture(uDye, vUv + vec2(uTexel.x, 0.0)).x;
  float hB = texture(uDye, vUv - vec2(0.0, uTexel.y)).x;
  float hT = texture(uDye, vUv + vec2(0.0, uTexel.y)).x;
  float p  = texture(uPressure, vUv).x;
  float pL = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float pR = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float pB = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float pT = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  vec2 vel = texture(uVelocity, vUv).xy;

  vec2 grad = vec2(hR - hL, hT - hB) * 6.0 + vec2(pR - pL, pT - pB) * 2.5;
  vec3 n = normalize(vec3(-grad, 0.35));

  vec3 lightDir = normalize(vec3(0.4, 0.6, 0.7));
  vec3 viewDir = vec3(0.0, 0.0, 1.0);
  float diff = max(dot(n, lightDir), 0.0);
  vec3 h = normalize(lightDir + viewDir);
  float spec = pow(max(dot(n, h), 0.0), 90.0);
  float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.0);

  // Fake refracted pool floor with caustic-ish tiles.
  vec2 floorUv = vUv + n.xy * 0.08;
  float tile = smoothstep(0.45, 0.5, abs(fract(floorUv.x * 12.0) - 0.5)) +
               smoothstep(0.45, 0.5, abs(fract(floorUv.y * 12.0) - 0.5));
  vec3 deep = vec3(0.02, 0.18, 0.35);
  vec3 shallow = vec3(0.10, 0.55, 0.75);
  vec3 floorCol = mix(deep, shallow, 0.5 + 0.5 * diff) + tile * 0.08;

  float dye = texture(uDye, vUv).x;
  vec3 col = floorCol;
  col += vec3(0.25, 0.45, 0.55) * clamp(dye, -1.0, 1.0) * 0.6;
  col += vec3(0.7, 0.85, 1.0) * fresnel * 0.35;
  col += vec3(1.0) * spec * 0.9;
  col += vec3(0.1, 0.2, 0.3) * clamp(p * 4.0, -0.3, 0.3);
  col += vec3(0.05) * min(length(vel) * 0.1, 0.5);
  frag = vec4(col, 1.0);
}`;

// Surface wave equation on the height field (x = height, y = vertical velocity, z = splash).
// h_tt = c^2 * Lap(h) - damping, with Neumann (reflecting) walls via clamped sampling.
// The boat hull is a solid: inside its footprint the surface is held below the hull bottom,
// and whatever water would have risen through it is written to z as "splash" for the spray.
export const WAVE = HEADER + `
uniform sampler2D uField;
uniform sampler2D uBoat;
uniform float uDt;
uniform float uC2;
uniform float uDamp;
uniform float uAspect;
uniform float uBoatSize;   // half-length of the hull footprint, in uv-y units
uniform float uDraft;      // keel depth in height units
void main() {
  vec2 f2 = texture(uField, vUv).xy;
  float hC = f2.x;
  float L = texture(uField, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uField, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uField, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uField, vUv + vec2(0.0, uTexel.y)).x;

  // Hull footprint geometry (ellipse in the boat's local frame, in uv-y units).
  vec4 b = texelFetch(uBoat, ivec2(0), 0);
  vec2 rel = vUv - b.xy; rel.x *= uAspect;
  vec2 fwd = vec2(cos(b.z), sin(b.z));
  vec2 side = vec2(-fwd.y, fwd.x);
  vec2 ax = vec2(1.0 / uBoatSize, 1.0 / (uBoatSize * 0.6));
  vec2 lc = vec2(dot(rel, fwd), dot(rel, side)) * ax;
  float r2 = dot(lc, lc);
  float splash = 0.0;
  if (r2 < 2.5) {
    // Neighbour cells inside the hull act as a wall (Neumann): use our own height for them.
    vec2 dx = vec2(uTexel.x * uAspect * fwd.x, uTexel.x * uAspect * side.x) * ax;
    vec2 dy = vec2(uTexel.y * fwd.y, uTexel.y * side.y) * ax;
    if (dot(lc - dx, lc - dx) < 1.0) L = hC;
    if (dot(lc + dx, lc + dx) < 1.0) R = hC;
    if (dot(lc - dy, lc - dy) < 1.0) B = hC;
    if (dot(lc + dy, lc + dy) < 1.0) T = hC;
  }
  float lap = L + R + B + T - 4.0 * hC;
  float v = clamp((f2.y + uC2 * lap * uDt) * uDamp, -1.0, 1.0);
  float h = clamp(hC + v * uDt, -1.0, 1.0);

  if (r2 < 2.5) {
    // Water level the boat floats at: mean of the surface at bow, stern and both beams.
    vec2 fuv = vec2(fwd.x / uAspect, fwd.y) * uBoatSize * 1.2;
    vec2 ruv = vec2(side.x / uAspect, side.y) * uBoatSize * 0.6 * 1.2;
    float hb = 0.25 * (texture(uField, b.xy + fuv).x + texture(uField, b.xy - fuv).x +
                       texture(uField, b.xy + ruv).x + texture(uField, b.xy - ruv).x);
    if (r2 < 1.0) {
      // Under the hull: the surface is the hull bottom, no free water.
      h = hb - uDraft * sqrt(1.0 - r2);
      v = 0.0;
    } else {
      // Ring around the hull: a crest rising above the waterline hits the topsides and is
      // thrown off as spray; the hull also pushes water down as it moves (wake).
      float ring = 1.0 - smoothstep(1.0, 1.6, r2);
      float excess = max(h - hb - 0.02, 0.0) * ring;
      // Spray when a crest rises above the waterline or the surface slams upward against the hull.
      splash = excess + max(v - 0.012, 0.0) * 3.0 * ring;
      h -= excess * 0.6;
      v = mix(v, min(v, 0.0), ring);
      h -= ring * min(b.w * 20.0, 0.15) * uDt;
    }
  }
  if (isnan(h) || isnan(v)) { h = 0.0; v = 0.0; }
  frag = vec4(h, v, splash, 1.0);
}`;

// ---- Spray particles (GPU). Two state textures: A = (x, y, z, life), B = (vx, vy, vz, seed).
export const PARTICLE_UPDATE = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
uniform sampler2D uA;
uniform sampler2D uB;
uniform sampler2D uField;
uniform sampler2D uBoat;
uniform float uDt;
uniform float uTime;
uniform float uAspect;
uniform float uHeightScale;
uniform float uBoatSize;
uniform float uSpawnRate;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec2 toUv(vec2 w) { return vec2(w.x / (2.0 * uAspect) + 0.5, w.y * 0.5 + 0.5); }
void main() {
  vec4 a = texture(uA, vUv);
  vec4 b = texture(uB, vUv);
  float seed = b.w;
  if (seed == 0.0) seed = hash(vUv * 17.0 + 0.3);
  if (a.w > 0.0) {
    vec3 vel = b.xyz;
    vel.z -= 0.0035 * uDt;
    vel *= pow(0.985, uDt);
    vec3 pos = a.xyz + vel * uDt;
    float life = a.w - uDt / 70.0;
    vec2 uv = toUv(pos.xy);
    float water = texture(uField, uv).x * uHeightScale;
    if (pos.z < water - 0.01 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) life = 0.0;
    oA = vec4(pos, life);
    oB = vec4(vel, seed);
    return;
  }
  // Dead: try to spawn at a random point on the hull rim where water is being thrown off.
  vec4 bt = texelFetch(uBoat, ivec2(0), 0);
  float r1 = hash(vec2(seed, uTime));
  float r2 = hash(vec2(uTime * 1.7, seed + 2.0));
  float r3 = hash(vec2(seed * 3.1, uTime + 5.0));
  float ang = r1 * 6.2831853;
  vec2 fwd = vec2(cos(bt.z), sin(bt.z));
  vec2 side = vec2(-fwd.y, fwd.x);
  vec2 local = vec2(cos(ang) * uBoatSize * 1.08, sin(ang) * uBoatSize * 0.6 * 1.08);
  vec2 off = fwd * local.x + side * local.y;
  vec2 uv = bt.xy + vec2(off.x / uAspect, off.y);
  vec3 fld = texture(uField, uv).xyz;
  float splash = fld.z;
  if (splash * uSpawnRate > r2 * 0.03) {
    vec3 pos = vec3((uv.x - 0.5) * 2.0 * uAspect, (uv.y - 0.5) * 2.0, fld.x * uHeightScale + 0.02);
    vec2 outward = normalize(off + 1e-5);
    float sp = min(splash * 1.5, 0.08);
    vec3 vel = vec3(outward * sp * (0.6 + r3), 0.035 + sp * 0.8 + r3 * 0.02);
    vel.xy += fwd * bt.w * 2.0 * uAspect; // carry the boat's motion
    oA = vec4(pos, 1.0);
    oB = vec4(vel, seed);
  } else {
    oA = vec4(0.0, 0.0, -10.0, 0.0);
    oB = vec4(0.0, 0.0, 0.0, seed);
  }
}`;

export const VERT_PARTICLE = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA;
uniform mat4 uVP;
uniform float uPointScale;
uniform int uTexW;
out float vLife;
void main() {
  ivec2 ij = ivec2(gl_VertexID % uTexW, gl_VertexID / uTexW);
  vec4 a = texelFetch(uA, ij, 0);
  vLife = a.w;
  gl_Position = uVP * vec4(a.xyz, 1.0);
  gl_PointSize = a.w > 0.0 ? uPointScale / gl_Position.w : 0.0;
}`;

export const FRAG_PARTICLE = `#version 300 es
precision highp float;
in float vLife;
out vec4 frag;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r = dot(d, d) * 4.0;
  if (r > 1.0) discard;
  float a = (1.0 - r) * smoothstep(0.0, 0.2, vLife) * 0.9;
  vec3 col = mix(vec3(0.75, 0.9, 1.0), vec3(1.0), 1.0 - r);
  frag = vec4(col, a);
}`;

export const VERT3D = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 aUv;
uniform sampler2D uHeight;
uniform mat4 uVP;
uniform float uAspect;
uniform float uHeightScale;
out vec2 vUv;
out vec3 vPos;
void main() {
  float h = texture(uHeight, aUv).x;
  vec3 p = vec3((aUv.x - 0.5) * 2.0 * uAspect, (aUv.y - 0.5) * 2.0, h * uHeightScale);
  vUv = aUv;
  vPos = p;
  gl_Position = uVP * vec4(p, 1.0);
}`;

export const FRAG3D = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
in vec3 vPos;
out vec4 frag;
uniform sampler2D uHeight;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
uniform vec3 uEye;
uniform float uAspect;
uniform float uHeightScale;

vec3 sky(vec3 d) {
  float t = clamp(d.z, 0.0, 1.0);
  vec3 horizon = vec3(0.75, 0.85, 0.95);
  vec3 zenith = vec3(0.10, 0.30, 0.65);
  vec3 c = mix(horizon, zenith, pow(t, 0.6));
  vec3 sun = normalize(vec3(0.4, 0.6, 0.7));
  c += vec3(1.0, 0.95, 0.85) * pow(max(dot(d, sun), 0.0), 400.0) * 2.0;
  return c;
}

void main() {
  float hL = texture(uHeight, vUv - vec2(uTexel.x, 0.0)).x;
  float hR = texture(uHeight, vUv + vec2(uTexel.x, 0.0)).x;
  float hB = texture(uHeight, vUv - vec2(0.0, uTexel.y)).x;
  float hT = texture(uHeight, vUv + vec2(0.0, uTexel.y)).x;
  float dx = 2.0 * uAspect * uTexel.x;
  float dy = 2.0 * uTexel.y;
  vec3 n = normalize(vec3(-(hR - hL) * uHeightScale / (2.0 * dx), -(hT - hB) * uHeightScale / (2.0 * dy), 1.0));

  vec3 V = normalize(uEye - vPos);
  vec3 Rv = reflect(-V, n);
  float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(n, V), 0.0), 5.0);

  // Refracted pool floor.
  vec3 Rf = refract(-V, n, 1.0 / 1.333);
  float depth = 0.5;
  float t = (-depth - vPos.z) / min(Rf.z, -0.05);
  vec3 floorP = vPos + Rf * t;
  vec2 fuv = vec2(floorP.x / (2.0 * uAspect) + 0.5, floorP.y * 0.5 + 0.5);
  float tile = smoothstep(0.46, 0.5, abs(fract(fuv.x * 8.0) - 0.5)) +
               smoothstep(0.46, 0.5, abs(fract(fuv.y * 8.0) - 0.5));
  vec3 floorCol = vec3(0.12, 0.45, 0.62) + tile * 0.10;
  float outside = step(0.0, -fuv.x) + step(1.0, fuv.x) + step(0.0, -fuv.y) + step(1.0, fuv.y);
  floorCol = mix(floorCol, vec3(0.05, 0.12, 0.2), clamp(outside, 0.0, 1.0));
  float absorb = exp(-length(Rf * t) * 1.2);
  vec3 water = vec3(0.02, 0.15, 0.3);
  vec3 refr = mix(water, floorCol, absorb);

  vec3 col = mix(refr, sky(Rv), fresnel);
  vec3 sun = normalize(vec3(0.4, 0.6, 0.7));
  vec3 H = normalize(sun + V);
  col += vec3(1.0) * pow(max(dot(n, H), 0.0), 120.0) * 0.8;
  float speed = length(texture(uVelocity, vUv).xy);
  col += vec3(0.15, 0.2, 0.2) * min(speed * 0.05, 0.4);
  frag = vec4(col, 1.0);
}`;

// 1x1 boat state texture: xy = position (uv), z = heading, w = speed (uv/frame).
export const BOAT_STATE = HEADER + `
uniform sampler2D uState;
uniform sampler2D uVelocity;
uniform float uDt;
uniform float uAspect;
void main() {
  vec4 s = texelFetch(uState, ivec2(0), 0);
  vec2 pos = s.xy;
  if (pos == vec2(0.0) || any(isnan(s)) || any(isinf(s))) { pos = vec2(0.5); s = vec4(pos, 0.0, 0.0); }
  vec2 vel = texture(uVelocity, pos).xy * uTexel; // uv per frame
  vec2 drift = vel * uDt * 0.5;
  vec2 np = clamp(pos + drift, vec2(0.06), vec2(0.94));
  float speed = length(np - pos);
  float heading = s.z;
  if (speed > 1e-5) {
    vec2 d = (np - pos); d.x *= uAspect;
    float target = atan(d.y, d.x);
    float diff = mod(target - heading + 3.14159265, 6.2831853) - 3.14159265;
    heading += diff * min(1.0, speed * 400.0) * 0.15;
  }
  frag = vec4(np, heading, speed);
}`;

export const VERT_BOAT = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec3 aPos;
in vec3 aCol;
uniform sampler2D uState;
uniform sampler2D uHeight;
uniform mat4 uVP;
uniform vec2 uTexel;
uniform float uAspect;
uniform float uHeightScale;
uniform float uScale;
out vec3 vCol;
out vec3 vWorld;
vec3 world(vec2 uv) {
  float h = texture(uHeight, uv).x;
  return vec3((uv.x - 0.5) * 2.0 * uAspect, (uv.y - 0.5) * 2.0, h * uHeightScale);
}
void main() {
  vec4 s = texelFetch(uState, ivec2(0), 0);
  vec2 uv = s.xy;
  float heading = s.z;
  vec2 fwdUv = vec2(cos(heading) / uAspect, sin(heading));
  vec2 rgtUv = vec2(sin(heading) / uAspect, -cos(heading));
  // Sample the surface just outside the hull footprint (bow, stern, both beams) for the attitude.
  float hl = uScale * 0.25 * 1.2;   // half-length in uv-y units, 20% outside the hull
  vec3 pf = world(uv + fwdUv * hl);
  vec3 pb = world(uv - fwdUv * hl);
  vec3 pr = world(uv + rgtUv * hl * 0.6);
  vec3 pl = world(uv - rgtUv * hl * 0.6);
  vec3 fwd = normalize(pf - pb);
  vec3 rgt = normalize(pr - pl);
  vec3 up = normalize(cross(fwd, rgt));
  if (up.z < 0.0) { up = -up; }
  rgt = normalize(cross(up, fwd));
  vec3 center = (pf + pb + pr + pl) * 0.25;
  vec3 p = center + (aPos.x * fwd + aPos.y * rgt + aPos.z * up) * uScale;
  vCol = aCol;
  vWorld = p;
  gl_Position = uVP * vec4(p, 1.0);
}`;

export const FRAG_BOAT = `#version 300 es
precision highp float;
in vec3 vCol;
in vec3 vWorld;
out vec4 frag;
uniform vec3 uEye;
void main() {
  vec3 n = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
  vec3 sun = normalize(vec3(0.4, 0.6, 0.7));
  vec3 V = normalize(uEye - vWorld);
  float diff = max(dot(n, sun), 0.0);
  vec3 H = normalize(sun + V);
  float spec = pow(max(dot(n, H), 0.0), 40.0) * 0.3;
  vec3 col = vCol * (0.35 + 0.65 * diff) + spec;
  frag = vec4(col, 1.0);
}`;
