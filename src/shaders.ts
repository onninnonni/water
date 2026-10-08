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
