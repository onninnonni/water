// Procedural soundscape: deep-sea music with low horn "yo-ho" moans, a quieter ocean bed, irregular
// water knocks and timber groans on the hull, and game effects (cannons, harpoon, serpents, thunder).
export interface AudioState {
  speed: number;   // 0..1 of top speed
  storm: number;   // 0..1
  rain: number;    // 0..1
  wind: number;    // 0..1
  heave: number;   // |vertical speed| m/s
  roll: number;    // |roll rate| rad/s
}

const NOTE = (n: number) => 440 * Math.pow(2, (n - 69) / 12);
const CHORDS = [
  [38, 45, 50, 53], [34, 41, 46, 50], [31, 38, 43, 46], [33, 40, 45, 50],
  [38, 45, 50, 53], [29, 36, 41, 45], [31, 38, 43, 46], [33, 40, 45, 49],
];
const vol = (dist: number, ref = 40) => 1 / (1 + Math.max(dist, 0) / ref);

export class SeaAudio {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private revIn!: GainNode;
  private musicBus!: GainNode;
  private white!: AudioBuffer;
  private brown!: AudioBuffer;
  private rumble!: GainNode; private rumbleF!: BiquadFilterNode;
  private wash!: GainNode; private washF!: BiquadFilterNode;
  private hull!: GainNode; private hullF!: BiquadFilterNode;
  private rainG!: GainNode;
  private wind!: GainNode; private windF!: BiquadFilterNode;
  private nextChord = 0; private chordIdx = 0;
  private nextPing = 0; private nextMoan = 0; private nextCreak = 0; private nextKnock = 0; private nextGroan = 0;
  private windWalk = 0.5;
  private storm = 0;
  private boss = 0; private bossTarget = 0;
  private nextBeat = 0; private beat = 0; private nextChoir = 0; private nextBrass = 0;
  private reelG: GainNode | null = null; private reelF: BiquadFilterNode | null = null;
  muted = false;

  get started() { return this.ctx !== null; }

  start() {
    if (this.ctx) { void this.ctx.resume(); return; }
    const ctx = new AudioContext();
    this.ctx = ctx;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -16; comp.ratio.value = 4; comp.attack.value = 0.01; comp.release.value = 0.3;
    comp.connect(ctx.destination);
    this.master = ctx.createGain(); this.master.gain.value = 0; this.master.connect(comp);
    this.master.gain.setTargetAtTime(0.9, ctx.currentTime, 1.5);

    const len = Math.floor(ctx.sampleRate * 5);
    const ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) { const d = ir.getChannelData(ch); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3); }
    const rev = ctx.createConvolver(); rev.buffer = ir;
    const revOut = ctx.createGain(); revOut.gain.value = 0.6;
    this.revIn = ctx.createGain();
    this.revIn.connect(rev); rev.connect(revOut); revOut.connect(this.master);
    this.musicBus = ctx.createGain(); this.musicBus.gain.value = 1.1; this.musicBus.connect(this.master);

    const n = ctx.sampleRate * 3;
    this.white = ctx.createBuffer(1, n, ctx.sampleRate);
    this.brown = ctx.createBuffer(1, n, ctx.sampleRate);
    const w = this.white.getChannelData(0), b = this.brown.getChannelData(0);
    let last = 0;
    for (let i = 0; i < n; i++) { w[i] = Math.random() * 2 - 1; last = (last + 0.02 * w[i]) / 1.02; b[i] = last * 3.5; }

    const loop = (buf: AudioBuffer) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, Math.random() * 2); return s; };
    const chain = (src: AudioNode, type: BiquadFilterType, f: number, q: number, toRev = 0) => {
      const fl = ctx.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
      const g = ctx.createGain(); g.gain.value = 0;
      src.connect(fl); fl.connect(g); g.connect(this.master);
      if (toRev) { const s = ctx.createGain(); s.gain.value = toRev; g.connect(s); s.connect(this.revIn); }
      return { fl, g };
    };
    ({ fl: this.rumbleF, g: this.rumble } = chain(loop(this.brown), 'lowpass', 200, 0.7, 0.3));
    ({ fl: this.washF, g: this.wash } = chain(loop(this.white), 'bandpass', 600, 0.5, 0.3));
    ({ fl: this.hullF, g: this.hull } = chain(loop(this.white), 'bandpass', 700, 1.2));
    const rainHp = ctx.createBiquadFilter(); rainHp.type = 'highpass'; rainHp.frequency.value = 2500;
    loop(this.white).connect(rainHp);
    ({ g: this.rainG } = chain(rainHp, 'lowpass', 7000, 0.5));
    ({ fl: this.windF, g: this.wind } = chain(loop(this.white), 'bandpass', 500, 8, 0.3));

    for (const [f, g] of [[NOTE(26), 0.1], [NOTE(38), 0.06], [NOTE(45), 0.03]] as [number, number][]) {
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.05 + Math.random() * 0.05;
      const lg = ctx.createGain(); lg.gain.value = f * 0.004; lfo.connect(lg); lg.connect(o.frequency); lfo.start();
      const og = ctx.createGain(); og.gain.value = g; o.connect(og); og.connect(this.musicBus);
      const s = ctx.createGain(); s.gain.value = 0.6; og.connect(s); s.connect(this.revIn);
      o.start();
    }
    const t = ctx.currentTime;
    this.nextChord = t + 1.5; this.nextPing = t + 9; this.nextMoan = t + 6; this.nextCreak = t + 4; this.nextKnock = t + 1; this.nextGroan = t + 8;
  }

  setMuted(m: boolean) {
    this.muted = m;
    if (this.ctx) this.master.gain.setTargetAtTime(m ? 0 : 0.9, this.ctx.currentTime, 0.2);
  }

  // ---- Music ----
  private chord() {
    const ctx = this.ctx!, t = ctx.currentTime, dur = 11;
    const notes = CHORDS[this.chordIdx++ % CHORDS.length];
    const filt = ctx.createBiquadFilter(); filt.type = 'lowpass'; filt.Q.value = 0.8;
    filt.frequency.setValueAtTime(300, t); filt.frequency.linearRampToValueAtTime(700 + 300 * this.storm, t + dur * 0.5); filt.frequency.linearRampToValueAtTime(300, t + dur + 5);
    const g = ctx.createGain(), peak = 0.05 + 0.03 * this.storm;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(peak, t + 4); g.gain.setValueAtTime(peak, t + dur); g.gain.linearRampToValueAtTime(0, t + dur + 6);
    filt.connect(g); g.connect(this.musicBus);
    const s = ctx.createGain(); s.gain.value = 1.2; g.connect(s); s.connect(this.revIn);
    for (const nn of notes) for (const det of [-6, 5]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = NOTE(nn); o.detune.value = det;
      o.connect(filt); o.start(t); o.stop(t + dur + 6.5);
    }
  }

  // A deep horn-choir moan: low detuned saws through "oo" formants, gliding up and falling away.
  private moan() {
    const ctx = this.ctx!, t = ctx.currentTime;
    const base = NOTE([26, 29, 31, 33, 24][Math.floor(Math.random() * 5)]);
    const dur = 5 + Math.random() * 4, peak = 0.07 + 0.08 * this.storm;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(peak, t + dur * 0.35); g.gain.linearRampToValueAtTime(peak * 0.7, t + dur * 0.7); g.gain.linearRampToValueAtTime(0, t + dur);
    const f1 = ctx.createBiquadFilter(); f1.type = 'bandpass'; f1.frequency.value = 330; f1.Q.value = 5;
    const f2 = ctx.createBiquadFilter(); f2.type = 'bandpass'; f2.frequency.value = 760; f2.Q.value = 7;
    const f2g = ctx.createGain(); f2g.gain.value = 0.45;
    f1.frequency.linearRampToValueAtTime(420, t + dur * 0.4); f1.frequency.linearRampToValueAtTime(300, t + dur);
    f1.connect(g); f2.connect(f2g); f2g.connect(g);
    g.connect(this.revIn); const dry = ctx.createGain(); dry.gain.value = 0.35; g.connect(dry); dry.connect(this.musicBus);
    for (const [mul, det] of [[1, -8], [1, 7], [2, 3], [1.5, -4]] as [number, number][]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.detune.value = det;
      o.frequency.setValueAtTime(base * mul * 0.94, t); o.frequency.linearRampToValueAtTime(base * mul, t + dur * 0.3); o.frequency.linearRampToValueAtTime(base * mul * 0.89, t + dur);
      const vib = ctx.createOscillator(); vib.frequency.value = 4.2 + Math.random(); const vg = ctx.createGain(); vg.gain.value = base * mul * 0.006;
      vib.connect(vg); vg.connect(o.frequency);
      o.connect(f1); o.connect(f2);
      o.start(t); vib.start(t); o.stop(t + dur + 0.1); vib.stop(t + dur + 0.1);
    }
  }

  private ping() {
    const ctx = this.ctx!, t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = NOTE([62, 69, 74, 57][Math.floor(Math.random() * 4)]);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.03, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0005, t + 3.5);
    o.connect(g); g.connect(this.revIn);
    o.start(t); o.stop(t + 3.6);
  }

  // ---- One-shots ----
  private burst(buf: AudioBuffer, type: BiquadFilterType, f: number, q: number, gain: number, attack: number, decay: number, delay = 0, rev = 0.3) {
    const ctx = this.ctx!, t = ctx.currentTime + delay;
    const s = ctx.createBufferSource(); s.buffer = buf;
    const fl = ctx.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(Math.max(gain, 0.0004), t + attack); g.gain.exponentialRampToValueAtTime(0.0003, t + attack + decay);
    s.connect(fl); fl.connect(g); g.connect(this.master);
    if (rev) { const r = ctx.createGain(); r.gain.value = rev; g.connect(r); r.connect(this.revIn); }
    s.start(t, Math.random() * 2, attack + decay + 0.1);
    return fl;
  }
  private tone(type: OscillatorType, f0: number, f1: number, gain: number, dur: number, delay = 0, rev = 0.2) {
    const ctx = this.ctx!, t = ctx.currentTime + delay;
    const o = ctx.createOscillator(); o.type = type;
    o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(Math.max(f1, 1), t + dur);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + 0.008); g.gain.exponentialRampToValueAtTime(0.0003, t + dur);
    o.connect(g); g.connect(this.master);
    if (rev) { const r = ctx.createGain(); r.gain.value = rev; g.connect(r); r.connect(this.revIn); }
    o.start(t); o.stop(t + dur + 0.05);
  }

  // Water hitting the hull: a knock, a slosh or a gurgle, with random character.
  private knock(i: number) {
    const k = Math.random();
    if (k < 0.45) {
      this.burst(this.brown, 'lowpass', 140 + Math.random() * 120, 1, 0.5 * i, 0.006, 0.18 + Math.random() * 0.2, 0, 0.15);
      this.burst(this.white, 'bandpass', 300 + Math.random() * 400, 1.5, 0.14 * i, 0.004, 0.12 + Math.random() * 0.15);
    } else if (k < 0.8) {
      const f = this.burst(this.white, 'bandpass', 900 + Math.random() * 500, 2.5, 0.12 * i, 0.03, 0.45 + Math.random() * 0.4, 0, 0.2);
      f.frequency.exponentialRampToValueAtTime(220 + Math.random() * 120, this.ctx!.currentTime + 0.6);
    } else {
      const n = 3 + Math.floor(Math.random() * 4);
      for (let j = 0; j < n; j++) { const f = 350 + Math.random() * 500; this.tone('sine', f, f * (1.6 + Math.random()), 0.035 * i, 0.06 + Math.random() * 0.05, j * (0.05 + Math.random() * 0.07), 0.1); }
    }
  }
  private creak(i: number) {
    const f = this.burst(this.white, 'bandpass', 220 + Math.random() * 160, 16, 0.13 * i, 0.12, 0.5 + Math.random() * 0.6, 0, 0.25);
    f.frequency.linearRampToValueAtTime(380 + Math.random() * 260, this.ctx!.currentTime + 0.7);
  }
  private groan(i: number) {
    const ctx = this.ctx!, t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'sawtooth';
    const f0 = 55 + Math.random() * 30; o.frequency.setValueAtTime(f0, t); o.frequency.linearRampToValueAtTime(f0 * (1.2 + Math.random() * 0.3), t + 1.4);
    const fl = ctx.createBiquadFilter(); fl.type = 'bandpass'; fl.frequency.value = 180; fl.Q.value = 6;
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.06 * i, t + 0.5); g.gain.linearRampToValueAtTime(0, t + 1.6);
    o.connect(fl); fl.connect(g); g.connect(this.master); const r = ctx.createGain(); r.gain.value = 0.3; g.connect(r); r.connect(this.revIn);
    o.start(t); o.stop(t + 1.7);
  }

  slap(i: number) {
    if (!this.ctx) return;
    this.burst(this.white, 'bandpass', 300 + Math.random() * 150, 0.9, 0.5 * i, 0.008, 0.4 + 0.3 * i);
    this.burst(this.brown, 'lowpass', 100, 0.7, 0.9 * i, 0.01, 0.5);
    this.burst(this.white, 'highpass', 2200, 0.5, 0.12 * i, 0.06, 0.9 + i, 0.1);
    this.creak(0.6 + 0.4 * i);
  }
  // The hull landing in a trough after a big wave: a deep boom, a roar of water, then the hiss.
  splashdown(i: number) {
    if (!this.ctx) return;
    this.tone('sine', 62, 28, 1.1 * i, 1.4, 0, 0.6);
    this.burst(this.brown, 'lowpass', 140, 0.8, 1.6 * i, 0.006, 1.3, 0, 0.7);
    this.burst(this.white, 'bandpass', 520, 0.7, 0.9 * i, 0.02, 1.6, 0.03, 0.6);
    this.burst(this.white, 'highpass', 1800, 0.4, 0.35 * i, 0.25, 2.6, 0.15, 0.5);
    this.creak(1); this.groan(0.6 + 0.4 * i);
  }
  crunch(i: number) {
    if (!this.ctx) return;
    this.burst(this.brown, 'lowpass', 260, 1, 1.3 * i, 0.01, 1.0);
    this.burst(this.white, 'bandpass', 1100, 2, 0.4 * i, 0.005, 0.3);
    this.creak(1); this.groan(1);
  }
  cannon(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 60);
    this.burst(this.brown, 'lowpass', 180, 0.8, 1.6 * v, 0.004, 1.6, 0, 0.8);
    this.burst(this.white, 'lowpass', 2500, 0.6, 0.7 * v, 0.002, 0.35, 0, 0.6);
    this.tone('sine', 90, 35, 0.9 * v, 0.6, 0, 0.5);
  }
  harpoon(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist);
    this.tone('triangle', 180, 90, 0.35 * v, 0.35, 0, 0.3);
    this.burst(this.white, 'bandpass', 1800, 3, 0.25 * v, 0.003, 0.25);
    this.burst(this.brown, 'lowpass', 300, 1, 0.5 * v, 0.004, 0.25);
  }
  splash(i: number, dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 80) * i;
    this.burst(this.white, 'bandpass', 700, 0.7, 0.5 * v, 0.01, 0.8, 0, 0.4);
    this.burst(this.brown, 'lowpass', 160, 1, 0.6 * v, 0.01, 0.6);
  }
  hit(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 80);
    this.burst(this.brown, 'lowpass', 220, 1.2, 1.1 * v, 0.004, 0.5, 0, 0.5);
    this.tone('sawtooth', 120, 50, 0.25 * v, 0.4, 0, 0.4);
  }
  roar(dist: number, pain = false) {
    if (!this.ctx) return;
    const ctx = this.ctx, t = ctx.currentTime, v = vol(dist, 90), dur = pain ? 1.2 : 2.4;
    const fl = ctx.createBiquadFilter(); fl.type = 'bandpass'; fl.frequency.value = 400; fl.Q.value = 2.5;
    fl.frequency.setValueAtTime(260, t); fl.frequency.linearRampToValueAtTime(pain ? 900 : 650, t + dur * 0.3); fl.frequency.linearRampToValueAtTime(200, t + dur);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.5 * v, t + 0.15); g.gain.linearRampToValueAtTime(0.35 * v, t + dur * 0.6); g.gain.linearRampToValueAtTime(0, t + dur);
    fl.connect(g); g.connect(this.master); const r = ctx.createGain(); r.gain.value = 0.7; g.connect(r); r.connect(this.revIn);
    const f0 = pain ? 95 : 55 + Math.random() * 20;
    for (const det of [0, 0.5, 1.01]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth';
      o.frequency.setValueAtTime(f0 * (1 + det * 0.01), t); o.frequency.linearRampToValueAtTime(f0 * (pain ? 1.6 : 1.3), t + dur * 0.3); o.frequency.linearRampToValueAtTime(f0 * 0.8, t + dur);
      const am = ctx.createOscillator(); am.frequency.value = 23 + det * 7; const ag = ctx.createGain(); ag.gain.value = f0 * 0.3; am.connect(ag); ag.connect(o.frequency);
      o.connect(fl); o.start(t); am.start(t); o.stop(t + dur + 0.05); am.stop(t + dur + 0.05);
    }
    this.burst(this.white, 'bandpass', 2400, 1.5, 0.1 * v, 0.1, dur * 0.8);
  }
  step(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 6);
    this.burst(this.brown, 'lowpass', 300 + Math.random() * 200, 1.5, 0.35 * v, 0.003, 0.09, 0, 0.05);
  }
  thunder(dist: number) {
    if (!this.ctx) return;
    const delay = Math.max(0.25, dist / 650);
    const v = 1.6 / (1 + dist / 1300);
    if (dist < 2600) this.burst(this.white, 'highpass', 1200, 0.5, 0.5 * v, 0.005, 0.3, delay, 0.6);
    const ctx = this.ctx, t = ctx.currentTime + delay, dur = 5 + Math.random() * 4;
    const s = ctx.createBufferSource(); s.buffer = this.brown; s.loop = true;
    const fl = ctx.createBiquadFilter(); fl.type = 'lowpass'; fl.frequency.setValueAtTime(700, t); fl.frequency.exponentialRampToValueAtTime(110, t + dur);
    const g = ctx.createGain();
    const curve = new Float32Array(64);
    for (let i = 0; i < 64; i++) { const x = i / 63; curve[i] = v * Math.pow(1 - x, 1.6) * (0.55 + 0.45 * Math.random()) * Math.min(1, x * 12); }
    g.gain.setValueAtTime(0, t); g.gain.setValueCurveAtTime(curve, t, dur);
    s.connect(fl); fl.connect(g); g.connect(this.master);
    const r = ctx.createGain(); r.gain.value = 0.8; g.connect(r); r.connect(this.revIn);
    s.start(t, Math.random() * 2); s.stop(t + dur + 0.1);
  }

  // ---- Boss battle ----
  setBoss(level: number) { this.bossTarget = level; }
  private drum(t: number, f: number, g: number, noise: number) {
    const ctx = this.ctx!;
    const o = ctx.createOscillator(); o.type = 'sine';
    o.frequency.setValueAtTime(f * 2.2, t); o.frequency.exponentialRampToValueAtTime(f, t + 0.12);
    const og = ctx.createGain(); og.gain.setValueAtTime(0, t); og.gain.linearRampToValueAtTime(g, t + 0.005); og.gain.exponentialRampToValueAtTime(0.0003, t + 0.7);
    o.connect(og); og.connect(this.master); const r = ctx.createGain(); r.gain.value = 0.35; og.connect(r); r.connect(this.revIn);
    o.start(t); o.stop(t + 0.75);
    if (noise > 0) {
      const sN = ctx.createBufferSource(); sN.buffer = this.brown;
      const fl = ctx.createBiquadFilter(); fl.type = 'lowpass'; fl.frequency.value = 900;
      const ng = ctx.createGain(); ng.gain.setValueAtTime(0, t); ng.gain.linearRampToValueAtTime(noise, t + 0.004); ng.gain.exponentialRampToValueAtTime(0.0003, t + 0.25);
      sN.connect(fl); fl.connect(ng); ng.connect(this.master); sN.start(t, Math.random() * 2, 0.3);
    }
  }
  // A huge male-choir "YO-HOOO": formant sweep from O to OO over very low detuned voices.
  private choir(t: number, notes: number[], dur: number, g: number) {
    const ctx = this.ctx!;
    const out = ctx.createGain();
    out.gain.setValueAtTime(0, t); out.gain.linearRampToValueAtTime(g, t + 0.25); out.gain.setValueAtTime(g, t + dur * 0.35);
    out.gain.linearRampToValueAtTime(g * 1.15, t + dur * 0.5); out.gain.linearRampToValueAtTime(0, t + dur);
    const f1 = ctx.createBiquadFilter(); f1.type = 'bandpass'; f1.Q.value = 4;
    f1.frequency.setValueAtTime(520, t); f1.frequency.linearRampToValueAtTime(560, t + dur * 0.3); f1.frequency.linearRampToValueAtTime(330, t + dur * 0.55);
    const f2 = ctx.createBiquadFilter(); f2.type = 'bandpass'; f2.Q.value = 6;
    f2.frequency.setValueAtTime(900, t); f2.frequency.linearRampToValueAtTime(720, t + dur * 0.55);
    const f2g = ctx.createGain(); f2g.gain.value = 0.5;
    f1.connect(out); f2.connect(f2g); f2g.connect(out);
    out.connect(this.revIn); const dry = ctx.createGain(); dry.gain.value = 0.6; out.connect(dry); dry.connect(this.musicBus);
    for (const nn of notes) for (const det of [-11, -3, 4, 12]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.detune.value = det;
      const f = NOTE(nn);
      o.frequency.setValueAtTime(f * 0.97, t); o.frequency.linearRampToValueAtTime(f, t + 0.2);
      o.frequency.setValueAtTime(f, t + dur * 0.4); o.frequency.linearRampToValueAtTime(f * 1.0595, t + dur * 0.5);
      const vib = ctx.createOscillator(); vib.frequency.value = 5 + Math.random(); const vg = ctx.createGain(); vg.gain.value = f * 0.008;
      vib.connect(vg); vg.connect(o.frequency);
      o.connect(f1); o.connect(f2); o.start(t); vib.start(t); o.stop(t + dur + 0.05); vib.stop(t + dur + 0.05);
    }
  }
  private brass(t: number, notes: number[], dur: number, g: number) {
    const ctx = this.ctx!;
    const fl = ctx.createBiquadFilter(); fl.type = 'lowpass'; fl.Q.value = 1.5;
    fl.frequency.setValueAtTime(200, t); fl.frequency.linearRampToValueAtTime(1400, t + dur * 0.4); fl.frequency.linearRampToValueAtTime(300, t + dur);
    const gg = ctx.createGain(); gg.gain.setValueAtTime(0, t); gg.gain.linearRampToValueAtTime(g, t + dur * 0.3); gg.gain.linearRampToValueAtTime(0, t + dur);
    fl.connect(gg); gg.connect(this.musicBus); const r = ctx.createGain(); r.gain.value = 0.6; gg.connect(r); r.connect(this.revIn);
    for (const nn of notes) for (const det of [-6, 6]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = NOTE(nn); o.detune.value = det;
      o.connect(fl); o.start(t); o.stop(t + dur + 0.05);
    }
  }
  private bossMusic(t: number) {
    const lvl = this.boss;
    // War drums at ~92 bpm: BOOM . . boom BOOM . boom .
    const step = 60 / 92 / 2;
    while (this.nextBeat < t + 0.2) {
      const b = this.beat++ % 16, at = this.nextBeat;
      const pat = [1, 0, 0, 0.5, 1, 0, 0.5, 0, 1, 0, 0, 0.5, 1, 0.4, 0.7, 0.9];
      if (pat[b] > 0) this.drum(at, b % 4 === 0 ? 42 : 55, 0.9 * pat[b] * lvl, b % 8 === 4 ? 0.5 * lvl : 0.15 * lvl);
      this.nextBeat += step;
    }
    if (t > this.nextChoir) {
      const prog = [[26, 33, 38], [22, 29, 34], [24, 31, 36], [21, 28, 33]];
      this.choir(t, prog[Math.floor(t / 7) % 4], 6.5, 0.16 * lvl);
      this.nextChoir = t + 6.6;
    }
    if (t > this.nextBrass) { this.brass(t + 0.1, [[38, 45, 50], [34, 41, 46], [36, 43, 48]][Math.floor(Math.random() * 3)], 3.2, 0.09 * lvl); this.nextBrass = t + 8 + Math.random() * 6; }
  }

  // ---- Combat sounds ----
  whoosh(dist: number) {
    if (!this.ctx) return;
    const f = this.burst(this.white, 'bandpass', 900, 1.5, 0.25 * vol(dist, 6), 0.06, 0.18, 0, 0.05);
    f.frequency.exponentialRampToValueAtTime(2600, this.ctx.currentTime + 0.18);
  }
  clash(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 8);
    for (const f of [1870, 2630, 3410]) this.tone('triangle', f, f * 0.98, 0.12 * v, 0.6, 0, 0.3);
    this.burst(this.white, 'highpass', 3000, 0.7, 0.3 * v, 0.002, 0.15);
  }
  cut(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 8);
    this.burst(this.brown, 'lowpass', 400, 1, 0.7 * v, 0.004, 0.25);
    this.burst(this.white, 'bandpass', 1500, 2, 0.2 * v, 0.004, 0.12);
  }
  rip(dist: number) {
    if (!this.ctx) return;
    const v = vol(dist, 60);
    const f = this.burst(this.white, 'bandpass', 600, 3, 0.6 * v, 0.01, 0.6, 0, 0.4);
    f.frequency.exponentialRampToValueAtTime(140, this.ctx.currentTime + 0.6);
    this.burst(this.brown, 'lowpass', 180, 1, 0.8 * v, 0.005, 0.5);
  }
  // Ratcheting winch while reeling the harpoon line in.
  reel(on: boolean) {
    if (!this.ctx) return;
    if (!this.reelG) {
      const ctx = this.ctx;
      const s = ctx.createBufferSource(); s.buffer = this.white; s.loop = true;
      this.reelF = ctx.createBiquadFilter(); this.reelF.type = 'bandpass'; this.reelF.frequency.value = 1400; this.reelF.Q.value = 4;
      const am = ctx.createOscillator(); am.type = 'square'; am.frequency.value = 14;
      const amG = ctx.createGain(); amG.gain.value = 0;
      this.reelG = ctx.createGain(); this.reelG.gain.value = 0;
      am.connect(amG.gain); amG.gain.value = 0.5;
      s.connect(this.reelF); this.reelF.connect(amG); amG.connect(this.reelG); this.reelG.connect(this.master);
      s.start(); am.start();
    }
    this.reelG.gain.setTargetAtTime(on ? 0.22 : 0, this.ctx.currentTime, 0.05);
  }

  update(a: AudioState, dt: number) {
    const ctx = this.ctx;
    if (!ctx) return;
    this.storm = a.storm;
    const t = ctx.currentTime, k = 0.25;
    this.boss += (this.bossTarget - this.boss) * Math.min(1, dt * 0.6);
    if (this.boss > 0.05) { if (this.nextBeat < t) this.nextBeat = t + 0.05; this.bossMusic(t); }
    if (t > this.nextChord && this.boss < 0.5) { this.chord(); this.nextChord = t + 11; }
    if (t > this.nextPing) { this.ping(); this.nextPing = t + 18 + Math.random() * 25; }
    if (t > this.nextMoan && this.boss < 0.5) { this.moan(); this.nextMoan = t + (a.storm > 0.5 ? 7 + Math.random() * 7 : a.storm > 0.2 ? 14 + Math.random() * 12 : 28 + Math.random() * 25); }
    // Irregular water knocks: more often with speed, heave and storm.
    const activity = 0.25 + a.speed * 0.8 + Math.min(a.heave, 3) * 0.35 + a.storm * 0.8;
    if (t > this.nextKnock) { this.knock(Math.min(0.35 + activity * 0.3, 1)); this.nextKnock = t + (0.3 + Math.random() * 2.2) / activity; }
    if (t > this.nextCreak && (a.roll > 0.05 || Math.random() < 0.3)) { this.creak(0.5 + Math.min(a.roll * 4, 0.5)); this.nextCreak = t + 1.5 + Math.random() * 4 / (0.5 + a.storm); }
    if (t > this.nextGroan) { this.groan(0.6 + a.storm * 0.5); this.nextGroan = t + 6 + Math.random() * 10 / (0.5 + a.storm); }
    const swell = 0.5 + 0.5 * Math.sin(t * 0.35) * Math.sin(t * 0.13 + 1);
    this.rumble.gain.setTargetAtTime(0.1 + 0.18 * a.storm * (0.6 + 0.4 * swell), t, k);
    this.rumbleF.frequency.setTargetAtTime(160 + 100 * a.storm, t, k);
    this.wash.gain.setTargetAtTime(0.01 + 0.03 * a.storm * swell + 0.012 * Math.min(a.heave, 3), t, k);
    this.washF.frequency.setTargetAtTime(450 + 350 * swell, t, k);
    this.hull.gain.setTargetAtTime(0.004 + a.speed * 0.05, t, 0.2);
    this.hullF.frequency.setTargetAtTime(500 + a.speed * 600, t, 0.2);
    this.rainG.gain.setTargetAtTime(a.rain * 0.05, t, 0.5);
    this.windWalk = Math.min(1, Math.max(0, this.windWalk + (Math.random() - 0.5) * dt * 0.8));
    this.wind.gain.setTargetAtTime((0.012 + 0.06 * a.wind) * (0.5 + this.windWalk), t, k);
    this.windF.frequency.setTargetAtTime(300 + 500 * this.windWalk + 250 * a.wind, t, k);
  }
}
