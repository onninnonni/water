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

// Surface wave equation on the height field (x = height, y = vertical velocity).
// h_tt = c^2 * Lap(h) - damping, with Neumann (reflecting) walls via clamped sampling.
export const WAVE = HEADER + `
uniform sampler2D uField;
uniform float uDt;
uniform float uC2;
uniform float uDamp;
void main() {
  vec2 f = texture(uField, vUv).xy;
  float L = texture(uField, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uField, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uField, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uField, vUv + vec2(0.0, uTexel.y)).x;
  float lap = L + R + B + T - 4.0 * f.x;
  float v = clamp((f.y + uC2 * lap * uDt) * uDamp, -2.0, 2.0);
  float h = clamp(f.x + v * uDt, -2.0, 2.0);
  frag = vec4(h, v, 0.0, 1.0);
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
