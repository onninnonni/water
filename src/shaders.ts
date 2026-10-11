// All GLSL for the game. Colours are authored in display space and pass through grade().
const H = `#version 300 es
precision highp float;
precision highp sampler2D;
`;

const COMMON = `
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uAmbTop;
uniform vec3 uAmbBot;
uniform vec3 uFogCol;
uniform float uFogDen;
uniform float uFlash;
uniform vec3 uFlashDir;
uniform float uTime;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }
vec3 flashLight(vec3 n){ return vec3(0.55,0.62,0.85) * uFlash * (0.35 + 0.65*max(dot(n, uFlashDir), 0.0)); }
vec3 lit(vec3 albedo, vec3 n){
  float d = max(dot(n, uSunDir), 0.0);
  vec3 amb = mix(uAmbBot, uAmbTop, n.z*0.5 + 0.5);
  return albedo * (amb + uSunCol*d + flashLight(n));
}
vec3 horizonCol(vec3 dir){
  float sd = max(dot(normalize(vec3(dir.xy, 0.0) + 1e-5), normalize(vec3(uSunDir.xy, 0.0))), 0.0);
  return uFogCol + uSunCol*0.22*pow(sd, 6.0) + vec3(0.22,0.25,0.32)*uFlash*0.45;
}
vec3 applyFog(vec3 c, vec3 p){
  vec3 d = p - uCamPos; float dist = length(d);
  return mix(c, horizonCol(d / max(dist, 1e-3)), 1.0 - exp(-dist * uFogDen));
}
vec3 grade(vec3 c){ c *= 1.2; c = (c*(2.51*c + 0.03)) / (c*(2.43*c + 0.59) + 0.14); return clamp(c, 0.0, 1.0); }
`;

const WAVES_GLSL = `
#define NW 12
uniform vec4 uWA[NW];
uniform vec4 uWB[NW];
uniform vec3 uIsland;
float stormAt(vec2 p){ return smoothstep(600.0, 3000.0, length(p - uIsland.xy)); }
float bayAt(vec2 p){ return 1.0 - smoothstep(uIsland.z - 140.0, uIsland.z - 30.0, length(p - uIsland.xy)); }
`;

export const VS_QUAD = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos*0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

// ---------- Sky: overcast storm clouds, thinner over the island, lit by lightning ----------
export const FS_SKY = H + COMMON + `
in vec2 vUv;
out vec4 frag;
uniform vec3 uFwd, uRight, uUp;
uniform float uTanHalf, uAspect;
uniform vec3 uSkyTop;
uniform vec3 uIsland;
uniform vec2 uWindDir;
uniform float uStorm;
float fbm(vec2 p){ float s = 0.0, a = 0.5; for (int i = 0; i < 5; i++){ s += a*vnoise(p); p = p*2.03 + vec2(1.7, 9.2); a *= 0.5; } return s; }
void main(){
  vec2 ndc = vUv*2.0 - 1.0;
  vec3 dir = normalize(uFwd + ndc.x*uTanHalf*uAspect*uRight + ndc.y*uTanHalf*uUp);
  vec3 col = horizonCol(dir);
  float up = max(dir.z, 0.0);
  col = mix(col, uSkyTop, pow(up, 0.55));
  float sd = max(dot(dir, uSunDir), 0.0);
  col += uSunCol*(0.35*pow(sd, 10.0) + 0.7*pow(sd, 90.0));
  col += vec3(0.07,0.08,0.09)*exp(-abs(dir.z)*22.0)*uStorm; // light leaking under the storm deck
  if (dir.z > 0.0) {
    // High cloud deck: domain-warped, dark undersides, bright torn edges; thinner over the island.
    float t = (1400.0 - uCamPos.z) / max(dir.z, 0.015);
    vec2 cp = uCamPos.xy + dir.xy*t;
    float cover = mix(0.42, 0.95, smoothstep(400.0, 3200.0, length(cp - uIsland.xy)));
    vec2 q = cp*0.0003 + vec2(uTime*0.010, uTime*0.004);
    q += 0.5*vec2(fbm(q*1.3 + 1.7), fbm(q*1.3 + 5.2));
    float n = fbm(q);
    float lo = 0.6 - 0.5*cover;
    float dens = smoothstep(lo, lo + 0.2, n);
    float shade = fbm(q*2.7 + 3.1);
    vec3 dark = uFogCol*0.38, light = uFogCol*1.5 + uSunCol*0.7*pow(sd, 2.0);
    float edge = 1.0 - smoothstep(0.0, 0.6, dens);
    vec3 cc = mix(dark, light, clamp(edge*0.8 + (1.0 - shade)*0.35 - 0.1, 0.0, 1.0));
    cc += vec3(0.7,0.75,0.95)*uFlash*(0.2 + 1.8*pow(max(dot(dir, uFlashDir), 0.0), 8.0))*(0.4 + 0.6*dens);
    float fade = smoothstep(0.0, 0.14, dir.z) * exp(-t*0.00004);
    col = mix(col, cc, dens*fade*0.97);
    // Low scud racing under the deck in the storm.
    float t2 = (520.0 - uCamPos.z) / max(dir.z, 0.015);
    vec2 cp2 = uCamPos.xy + dir.xy*t2;
    vec2 q2 = cp2*0.0011 - uWindDir*uTime*0.03;
    float scud = smoothstep(0.52, 0.78, fbm(q2 + 0.4*fbm(q2*2.0))) * uStorm;
    vec3 sc = uFogCol*0.55 + vec3(0.6,0.65,0.85)*uFlash*0.5;
    col = mix(col, sc, scud*0.8*smoothstep(0.0, 0.1, dir.z)*exp(-t2*0.00012));
  }
  frag = vec4(grade(col), 1.0);
}`;

// ---------- Ocean ----------
export const VS_OCEAN = `#version 300 es
precision highp float;
in vec3 aGrid; // offset x, offset y, local grid spacing
uniform mat4 uVP;
uniform vec2 uCenter;
` + WAVES_GLSL + `
out vec3 vPos;
out vec3 vN;
out float vCrest;
out vec2 vP0;
out float vStorm;
void main(){
  vec2 p0 = uCenter + aGrid.xy;
  float sp = aGrid.z;
  float s = stormAt(p0), bay = bayAt(p0);
  float ml = 0.04 + 1.6*s, ms = (0.22 + 0.78*s)*(1.0 - 0.55*bay);
  vec3 D = vec3(0.0); vec3 N = vec3(0.0, 0.0, 1.0); float crest = 0.0;
  for (int i = 0; i < NW; i++){
    vec4 A = uWA[i], B = uWB[i];
    float amp = A.w * mix(ms, ml, B.z) * (1.0 - smoothstep(B.w/6.0, B.w/3.0, sp));
    float th = A.z*dot(A.xy, p0) + B.y;
    float c = cos(th), sn = sin(th), qa = B.x*amp, wa = A.z*amp;
    D.xy += qa*A.xy*c; D.z += amp*sn;
    N.xy -= A.xy*wa*c; N.z -= B.x*wa*sn; crest += B.x*wa*sn;
  }
  vPos = vec3(p0 + D.xy, D.z); vN = N; vCrest = crest; vP0 = p0; vStorm = s;
  gl_Position = uVP*vec4(vPos, 1.0);
}`;

export const FS_OCEAN = H + COMMON + `
in vec3 vPos;
in vec3 vN;
in float vCrest;
in vec2 vP0;
in float vStorm;
out vec4 frag;
uniform sampler2D uWake;
uniform vec3 uWakeRect;  // origin x, origin y, size
uniform sampler2D uShore;
uniform vec3 uShoreRect; // min x, min y, size
uniform vec3 uSkyTop;
uniform vec2 uWind;
vec3 skyReflect(vec3 r){
  vec3 c = mix(horizonCol(r), uSkyTop, pow(max(r.z, 0.0), 0.55));
  float sd = max(dot(r, uSunDir), 0.0);
  return c + uSunCol*(0.3*pow(sd, 8.0)) + vec3(0.6,0.65,0.85)*uFlash*0.6;
}
void main(){
  vec3 n = normalize(vN);
  vec3 toCam = uCamPos - vPos; float dist = length(toCam); vec3 V = toCam / dist;
  // Wind ripples and chop, stronger in the storm.
  float det = 1.0 - smoothstep(25.0, 260.0, dist);
  vec2 rp = vPos.xy, w = uWind, wp = vec2(-w.y, w.x);
  vec2 rg = vec2(0.0);
  rg += (w*2.1 + wp*0.7)*cos(dot(rp, w*2.1 + wp*0.7) - uTime*4.6)*0.012;
  rg += (w*3.3 - wp*1.6)*cos(dot(rp, w*3.3 - wp*1.6) - uTime*5.8 + 1.7)*0.008;
  rg += (w*5.7 + wp*3.1)*cos(dot(rp, w*5.7 + wp*3.1) - uTime*7.5 + 0.4)*0.005;
  rg += (-w*2.4 + wp*7.9)*cos(dot(rp, -w*2.4 + wp*7.9) - uTime*9.1 + 2.9)*0.003;
  n = normalize(vec3(n.xy - rg*det*(0.6 + 1.6*vStorm), n.z));

  // Water depth from the baked shore map.
  vec2 suv = (vP0 - uShoreRect.xy) / uShoreRect.z;
  float ground = (suv.x > 0.0 && suv.y > 0.0 && suv.x < 1.0 && suv.y < 1.0) ? texture(uShore, suv).r*40.0 - 20.0 : -40.0;
  float depth = vPos.z - ground;
  float shallow = 1.0 - smoothstep(1.0, 20.0, depth);

  vec3 deep = mix(vec3(0.02,0.07,0.09), vec3(0.012,0.04,0.055), vStorm);
  vec3 water = mix(deep, vec3(0.05,0.2,0.19), shallow);
  vec3 body = water * (uAmbTop*1.6 + uSunCol*0.5 + flashLight(n));
  // Light glowing through thin crests.
  float sss = clamp(vPos.z*0.16, 0.0, 1.0) * (0.35 + 0.65*pow(max(dot(-V, uSunDir)*0.5 + 0.5, 0.0), 3.0));
  body += vec3(0.02,0.16,0.13)*sss*(uAmbTop*2.0 + uSunCol);

  vec3 R = reflect(-V, n); R.z = abs(R.z);
  float fres = 0.02 + 0.98*pow(1.0 - max(dot(n, V), 0.0), 5.0);
  vec3 col = mix(body, skyReflect(R), fres*0.85);
  vec3 Hh = normalize(uSunDir + V);
  col += uSunCol*pow(max(dot(n, Hh), 0.0), 220.0)*1.4;

  // Foam: breaking crests, the boat's wake, surf on the shore, wind streaks.
  float fn = vnoise(vP0*0.33 + uTime*0.15)*0.6 + vnoise(vP0*1.4 - uTime*0.3)*0.4;
  float crest = smoothstep(0.24, 0.5, vCrest);
  vec2 wuv = (vPos.xy - uWakeRect.xy) / uWakeRect.z + 0.5;
  float wake = (wuv.x > 0.0 && wuv.y > 0.0 && wuv.x < 1.0 && wuv.y < 1.0) ? texture(uWake, wuv).r : 0.0;
  float bub = vnoise(vPos.xy*2.2 + uTime*0.5)*0.6 + vnoise(vPos.xy*5.3 - uTime*0.8)*0.4;
  wake *= smoothstep(0.25, 0.75, bub + wake*0.45);
  float surf = (1.0 - smoothstep(0.0, 3.5, depth)) * (0.55 + 0.45*sin(uTime*1.3 - depth*1.7));
  vec2 sp = vec2(dot(vP0, w), dot(vP0, wp));
  float streak = smoothstep(0.62, 0.9, vnoise(vec2(sp.x*0.012, sp.y*0.12))) * smoothstep(0.35, 1.0, vStorm) * 0.6;
  float foam = clamp(crest*(0.45 + fn) + wake*(0.55 + 0.6*fn) + surf*fn*1.5 + streak*fn, 0.0, 1.0);
  vec3 foamCol = vec3(0.82,0.86,0.88) * (uAmbTop*1.9 + uSunCol*0.7 + flashLight(vec3(0,0,1)));
  col = mix(col, foamCol, foam*0.92);

  frag = vec4(grade(applyFog(col, vPos)), 1.0);
}`;

// ---------- Solid objects (land, boat, buoys) ----------
export const VS_SOLID = `#version 300 es
in vec3 aPos;
in vec3 aCol;
uniform mat4 uVP;
uniform mat4 uModel;
out vec3 vW;
out vec3 vCol;
void main(){ vec4 w = uModel*vec4(aPos, 1.0); vW = w.xyz; vCol = aCol; gl_Position = uVP*w; }`;

export const FS_SOLID = H + COMMON + `
in vec3 vW;
in vec3 vCol;
out vec4 frag;
uniform float uWet;
void main(){
  if (vCol.r > 1.5) { frag = vec4(grade(applyFog((vCol - 2.0)*1.7, vW)), 1.0); return; }
  vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
  if (dot(n, uCamPos - vW) < 0.0) n = -n;
  vec3 c = lit(vCol, n);
  c *= mix(1.0, mix(0.55, 1.0, smoothstep(-0.5, 3.0, vW.z)), uWet);
  frag = vec4(grade(applyFog(c, vW)), 1.0);
}`;

// ---------- Spray droplets ----------
export const VS_SPRAY = `#version 300 es
in vec4 aP; // xyz, life
uniform mat4 uVP;
uniform float uPx;
out float vLife;
out vec3 vW;
void main(){
  vW = aP.xyz; vLife = aP.w;
  vec4 cp = uVP*vec4(aP.xyz, 1.0);
  gl_Position = cp;
  gl_PointSize = aP.w > 0.0 ? clamp(uPx*0.07/max(cp.w, 0.1), 1.2, 22.0) : 0.0;
}`;

export const FS_SPRAY = H + COMMON + `
in float vLife;
in vec3 vW;
out vec4 frag;
void main(){
  vec2 d = gl_PointCoord - 0.5; float r = dot(d, d)*4.0;
  if (r > 1.0) discard;
  vec3 c = vec3(0.85,0.9,0.95)*(uAmbTop*1.9 + uSunCol*0.6 + flashLight(vec3(0,0,1)));
  float near = smoothstep(1.2, 5.0, length(vW - uCamPos));
  frag = vec4(grade(applyFog(c, vW)), (1.0 - r)*clamp(vLife*1.5, 0.0, 1.0)*0.6*near);
}`;

// ---------- Rain streaks around the camera ----------
export const VS_RAIN = `#version 300 es
precision highp float;
uniform mat4 uVP;
uniform vec3 uCamPos;
uniform vec3 uCamVel;
uniform vec3 uRainVel;
uniform float uTime;
uniform float uBox;
uniform mat4 uBoatInv;
uniform float uMask;
out float vA;
float h11(float p){ p = fract(p*0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
void main(){
  int id = gl_VertexID / 2; int e = gl_VertexID - id*2;
  float f = float(id);
  vec3 base = vec3(h11(f*0.731 + 0.1), h11(f*1.37 + 0.7), h11(f*2.11 + 0.3))*uBox;
  vec3 rel = mod(base + uRainVel*uTime - uCamPos, uBox) - 0.5*uBox;
  vec3 w = uCamPos + rel;
  vA = 1.0 - smoothstep(0.25*uBox, 0.5*uBox, length(rel));
  if (uMask > 0.5) {
    vec3 l = (uBoatInv*vec4(w, 1.0)).xyz;
    if (l.x > -3.75 && l.x < 0.5 && abs(l.y) < 1.9 && l.z > 0.8 && l.z < 4.2) vA = 0.0;
  }
  if (e == 1) w -= (uRainVel - uCamVel)*0.045;
  gl_Position = uVP*vec4(w, 1.0);
}`;

export const FS_RAIN = H + COMMON + `
in float vA;
out vec4 frag;
uniform float uAmt;
void main(){
  vec3 c = vec3(0.7,0.75,0.82)*(uAmbTop*2.2 + flashLight(vec3(0,0,1))*1.5);
  frag = vec4(grade(c), vA*0.32*uAmt);
}`;

// ---------- Additive glows: lamps, buoy lights, lightning ----------
export const VS_GLOW = `#version 300 es
in vec3 aCenter;
in vec3 aCorner; // corner.xy, size
in vec4 aCol;
uniform mat4 uVP;
uniform vec3 uCamRight, uCamUp;
out vec2 vC;
out vec4 vCol;
void main(){
  vC = aCorner.xy; vCol = aCol;
  vec3 w = aCenter + (uCamRight*aCorner.x + uCamUp*aCorner.y)*aCorner.z;
  gl_Position = uVP*vec4(w, 1.0);
}`;

export const FS_GLOW = H + `
in vec2 vC;
in vec4 vCol;
out vec4 frag;
void main(){
  float r2 = dot(vC, vC);
  float a = exp(-r2*5.0) + 0.25*exp(-r2*1.2);
  frag = vec4(vCol.rgb*vCol.a*a, 1.0);
}`;

export const VS_BOLT = `#version 300 es
in vec3 aPos;
in float aU;
uniform mat4 uVP;
out float vU;
void main(){ vU = aU; gl_Position = uVP*vec4(aPos, 1.0); }`;

export const FS_BOLT = H + `
in float vU;
out vec4 frag;
uniform float uI;
void main(){
  float x = vU*2.0 - 1.0;
  float a = exp(-x*x*14.0) + 0.35*exp(-x*x*2.5);
  frag = vec4(vec3(0.82,0.86,1.0)*a*uI, 1.0);
}`;

// ---------- Lighthouse beam (additive cone) ----------
export const VS_BEAM = `#version 300 es
in vec4 aP; // xyz, t along the beam
uniform mat4 uVP;
uniform mat4 uModel;
out float vT;
out vec3 vW;
void main(){ vT = aP.w; vec4 w = uModel*vec4(aP.xyz, 1.0); vW = w.xyz; gl_Position = uVP*w; }`;

export const FS_BEAM = H + `
in float vT;
in vec3 vW;
out vec4 frag;
uniform vec3 uCamPos;
uniform float uFogDen;
uniform float uI;
void main(){
  float a = pow(1.0 - vT, 1.6) * exp(-length(vW - uCamPos)*uFogDen*0.6) * uI;
  frag = vec4(vec3(1.0,0.88,0.65)*a*0.075, 1.0);
}`;

// ---------- Wake foam map (ping-pong, world-aligned around the boat) ----------
export const FS_WAKE = H + `
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uTexel;
uniform vec4 uSeg[3];   // a.xy, b.xy in texture uv
uniform vec2 uSegP[3];  // width (uv), intensity
uniform vec3 uBlob;     // centre uv, radius uv
uniform float uBlobI;
float segDist(vec2 p, vec2 a, vec2 b){ vec2 pa = p - a, ba = b - a; float h = clamp(dot(pa, ba)/max(dot(ba, ba), 1e-9), 0.0, 1.0); return length(pa - ba*h); }
void main(){
  float c = texture(uSrc, vUv).r;
  float nb = texture(uSrc, vUv + vec2(uTexel.x, 0)).r + texture(uSrc, vUv - vec2(uTexel.x, 0)).r
           + texture(uSrc, vUv + vec2(0, uTexel.y)).r + texture(uSrc, vUv - vec2(0, uTexel.y)).r;
  float v = mix(c, nb*0.25, 0.12)*0.99 - 0.0015;
  for (int i = 0; i < 3; i++){ float d = segDist(vUv, uSeg[i].xy, uSeg[i].zw); v += uSegP[i].y*exp(-d*d/(uSegP[i].x*uSegP[i].x)); }
  float db = length(vUv - uBlob.xy); v += uBlobI*exp(-db*db/(uBlob.z*uBlob.z));
  frag = vec4(clamp(v, 0.0, 1.0), 0.0, 0.0, 1.0);
}`;

export const FS_SHIFT = H + `
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uShift;
void main(){
  vec2 s = vUv + uShift;
  frag = (s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) ? vec4(0.0) : texture(uSrc, s);
}`;
