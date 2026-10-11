// Procedural storm soundscape: deep-sea music, ocean, wind, rain, diesel, hull slaps, thunder.
export interface AudioState {
  speed: number;      // 0..1 of top speed
  rpm: number;        // -0.5..1
  storm: number;      // 0..1
  rain: number;       // 0..1
  wind: number;       // 0..1
  heave: number;      // |vertical speed| m/s
  roll: number;       // |roll rate| rad/s
}

const NOTE = (n: number) => 440 * Math.pow(2, (n - 69) / 12);
// Dark D-minor progression, low voicings (MIDI notes).
const CHORDS = [
  [38, 45, 50, 53], // Dm
  [34, 41, 46, 50], // Bb
  [31, 38, 43, 46], // Gm
  [33, 40, 45, 50], // Asus4
  [38, 45, 50, 53], // Dm
  [29, 36, 41, 45], // F
  [31, 38, 43, 46], // Gm
  [33, 40, 45, 49], // A
];

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
  private gust!: GainNode; private gustF!: BiquadFilterNode;
  private eng!: OscillatorNode; private engSub!: OscillatorNode; private engG!: GainNode; private engF!: BiquadFilterNode;
  private nextChord = 0; private chordIdx = 0;
  private nextPing = 0; private nextWhale = 0; private nextCreak = 0;
  private windWalk = 0.5; private gustWalk = 0.3;
  muted = false;

  get started() { return this.ctx !== null; }

  start() {
    if (this.ctx) { void this.ctx.resume(); return; }
    const ctx = new AudioContext();
    this.ctx = ctx;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18; comp.ratio.value = 4; comp.attack.value = 0.01; comp.release.value = 0.3;
    comp.connect(ctx.destination);
    this.master = ctx.createGain(); this.master.gain.value = 0; this.master.connect(comp);
    this.master.gain.setTargetAtTime(0.9, ctx.currentTime, 1.5);

    // Reverb: generated stereo impulse response.
    const len = Math.floor(ctx.sampleRate * 4.5);
    const ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = ir.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3.2);
    }
    const rev = ctx.createConvolver(); rev.buffer = ir;
    const revOut = ctx.createGain(); revOut.gain.value = 0.55;
    this.revIn = ctx.createGain();
    this.revIn.connect(rev); rev.connect(revOut); revOut.connect(this.master);
    this.musicBus = ctx.createGain(); this.musicBus.gain.value = 0.85;
    this.musicBus.connect(this.master);

    // Noise buffers.
    const n = ctx.sampleRate * 3;
    this.white = ctx.createBuffer(1, n, ctx.sampleRate);
    this.brown = ctx.createBuffer(1, n, ctx.sampleRate);
    const w = this.white.getChannelData(0), b = this.brown.getChannelData(0);
    let last = 0;
    for (let i = 0; i < n; i++) { w[i] = Math.random() * 2 - 1; last = (last + 0.02 * w[i]) / 1.02; b[i] = last * 3.5; }

    const loop = (buf: AudioBuffer) => { const s = ctx.createBufferSource(); s.buffer = buf; s.loop = true; s.start(0, Math.random() * 2); return s; };
    const chain = (src: AudioNode, type: BiquadFilterType, f: number, q: number, gain: number, toRev = 0) => {
      const fl = ctx.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
      const g = ctx.createGain(); g.gain.value = gain;
      src.connect(fl); fl.connect(g); g.connect(this.master);
      if (toRev) { const s = ctx.createGain(); s.gain.value = toRev; g.connect(s); s.connect(this.revIn); }
      return { fl, g };
    };
    ({ fl: this.rumbleF, g: this.rumble } = chain(loop(this.brown), 'lowpass', 220, 0.7, 0.3, 0.3));
    ({ fl: this.washF, g: this.wash } = chain(loop(this.white), 'bandpass', 650, 0.5, 0.05, 0.4));
    ({ fl: this.hullF, g: this.hull } = chain(loop(this.white), 'bandpass', 900, 1.1, 0));
    const rainHp = ctx.createBiquadFilter(); rainHp.type = 'highpass'; rainHp.frequency.value = 2200;
    loop(this.white).connect(rainHp);
    ({ g: this.rainG } = chain(rainHp, 'lowpass', 8000, 0.5, 0));
    ({ fl: this.windF, g: this.wind } = chain(loop(this.white), 'bandpass', 500, 7, 0, 0.3));
    ({ fl: this.gustF, g: this.gust } = chain(loop(this.white), 'bandpass', 1800, 3, 0, 0.2));

    // Diesel: muffled saw throb.
    this.engF = ctx.createBiquadFilter(); this.engF.type = 'lowpass'; this.engF.frequency.value = 160; this.engF.Q.value = 3;
    this.engG = ctx.createGain(); this.engG.gain.value = 0;
    this.eng = ctx.createOscillator(); this.eng.type = 'sawtooth'; this.eng.frequency.value = 30;
    this.engSub = ctx.createOscillator(); this.engSub.type = 'sine'; this.engSub.frequency.value = 15;
    this.eng.connect(this.engF); this.engSub.connect(this.engF); this.engF.connect(this.engG); this.engG.connect(this.master);
    this.eng.start(); this.engSub.start();

    // Deep drone under everything.
    for (const [f, g] of [[NOTE(26), 0.09], [NOTE(38), 0.05], [NOTE(45), 0.025]] as [number, number][]) {
      const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = f;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.05 + Math.random() * 0.05;
      const lg = ctx.createGain(); lg.gain.value = f * 0.004; lfo.connect(lg); lg.connect(o.frequency); lfo.start();
      const og = ctx.createGain(); og.gain.value = g; o.connect(og); og.connect(this.musicBus);
      const s = ctx.createGain(); s.gain.value = 0.6; og.connect(s); s.connect(this.revIn);
      o.start();
    }
    const t = ctx.currentTime;
    this.nextChord = t + 1.5; this.nextPing = t + 9; this.nextWhale = t + 25; this.nextCreak = t + 6;
  }

  setMuted(m: boolean) {
    this.muted = m;
    if (this.ctx) this.master.gain.setTargetAtTime(m ? 0 : 0.9, this.ctx.currentTime, 0.2);
  }

  private chord() {
    const ctx = this.ctx!, t = ctx.currentTime, dur = 11;
    const notes = CHORDS[this.chordIdx++ % CHORDS.length];
    const filt = ctx.createBiquadFilter(); filt.type = 'lowpass'; filt.Q.value = 0.8;
    filt.frequency.setValueAtTime(300, t); filt.frequency.linearRampToValueAtTime(750, t + dur * 0.5); filt.frequency.linearRampToValueAtTime(300, t + dur + 5);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.045, t + 4); g.gain.setValueAtTime(0.045, t + dur); g.gain.linearRampToValueAtTime(0, t + dur + 6);
    filt.connect(g); g.connect(this.musicBus);
    const s = ctx.createGain(); s.gain.value = 1.2; g.connect(s); s.connect(this.revIn);
    for (const nn of notes) for (const det of [-6, 5]) {
      const o = ctx.createOscillator(); o.type = 'sawtooth'; o.frequency.value = NOTE(nn); o.detune.value = det;
      o.connect(filt); o.start(t); o.stop(t + dur + 6.5);
    }
  }

  private ping() {
    const ctx = this.ctx!, t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'sine'; o.frequency.value = NOTE([62, 69, 74, 57][Math.floor(Math.random() * 4)]);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.03, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0005, t + 3.5);
    o.connect(g); g.connect(this.revIn); const d = ctx.createGain(); d.gain.value = 0.3; g.connect(d); d.connect(this.musicBus);
    o.start(t); o.stop(t + 3.6);
  }

  private whale() {
    const ctx = this.ctx!, t = ctx.currentTime;
    const o = ctx.createOscillator(); o.type = 'sine';
    const f0 = 140 + Math.random() * 60;
    o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f0 * 0.62, t + 3.2);
    const vib = ctx.createOscillator(); vib.frequency.value = 4.5; const vg = ctx.createGain(); vg.gain.value = 3; vib.connect(vg); vg.connect(o.frequency);
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.022, t + 1); g.gain.linearRampToValueAtTime(0, t + 3.4);
    o.connect(g); g.connect(this.revIn);
    o.start(t); vib.start(t); o.stop(t + 3.5); vib.stop(t + 3.5);
  }

  private burst(buf: AudioBuffer, type: BiquadFilterType, f: number, q: number, gain: number, attack: number, decay: number, delay = 0, rev = 0.3) {
    const ctx = this.ctx!, t = ctx.currentTime + delay;
    const s = ctx.createBufferSource(); s.buffer = buf;
    const fl = ctx.createBiquadFilter(); fl.type = type; fl.frequency.value = f; fl.Q.value = q;
    const g = ctx.createGain(); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + attack); g.gain.exponentialRampToValueAtTime(0.0003, t + attack + decay);
    s.connect(fl); fl.connect(g); g.connect(this.master);
    if (rev) { const r = ctx.createGain(); r.gain.value = rev; g.connect(r); r.connect(this.revIn); }
    s.start(t, Math.random() * 2, attack + decay + 0.1);
    return fl;
  }

  slap(i: number) {
    if (!this.ctx) return;
    this.burst(this.white, 'bandpass', 380 + Math.random() * 150, 0.9, 0.55 * i, 0.008, 0.35 + 0.3 * i);
    this.burst(this.brown, 'lowpass', 120, 0.7, 0.8 * i, 0.01, 0.4);
    this.burst(this.white, 'highpass', 2500, 0.5, 0.25 * i, 0.05, 0.9 + i, 0.08); // spray hiss
  }
  crunch(i: number) {
    if (!this.ctx) return;
    this.burst(this.brown, 'lowpass', 300, 1, 1.2 * i, 0.01, 0.8);
    this.burst(this.white, 'bandpass', 1300, 2, 0.35 * i, 0.005, 0.25);
  }
  thunder(dist: number) {
    if (!this.ctx) return;
    const delay = Math.max(0.25, dist / 650);
    const vol = 1.6 / (1 + dist / 1300);
    if (dist < 2600) this.burst(this.white, 'highpass', 1200, 0.5, 0.5 * vol, 0.005, 0.3, delay, 0.6);
    const ctx = this.ctx, t = ctx.currentTime + delay, dur = 5 + Math.random() * 4;
    const s = ctx.createBufferSource(); s.buffer = this.brown; s.loop = true;
    const fl = ctx.createBiquadFilter(); fl.type = 'lowpass'; fl.frequency.setValueAtTime(700, t); fl.frequency.exponentialRampToValueAtTime(110, t + dur);
    const g = ctx.createGain();
    const curve = new Float32Array(64);
    for (let i = 0; i < 64; i++) { const x = i / 63; curve[i] = vol * Math.pow(1 - x, 1.6) * (0.55 + 0.45 * Math.random()) * Math.min(1, x * 12); }
    g.gain.setValueAtTime(0, t); g.gain.setValueCurveAtTime(curve, t, dur);
    s.connect(fl); fl.connect(g); g.connect(this.master);
    const r = ctx.createGain(); r.gain.value = 0.8; g.connect(r); r.connect(this.revIn);
    s.start(t, Math.random() * 2); s.stop(t + dur + 0.1);
  }

  update(a: AudioState, dt: number) {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime, k = 0.25;
    if (t > this.nextChord) { this.chord(); this.nextChord = t + 11; }
    if (t > this.nextPing) { this.ping(); this.nextPing = t + 16 + Math.random() * 22; }
    if (t > this.nextWhale) { this.whale(); this.nextWhale = t + 45 + Math.random() * 50; }
    if (a.roll > 0.12 && t > this.nextCreak) {
      const f = this.burst(this.white, 'bandpass', 260 + Math.random() * 120, 14, 0.12, 0.08, 0.5, 0, 0.2);
      f.frequency.linearRampToValueAtTime(420 + Math.random() * 200, t + 0.5);
      this.nextCreak = t + 2.5 + Math.random() * 4;
    }
    const swell = 0.5 + 0.5 * Math.sin(t * 0.35) * Math.sin(t * 0.13 + 1);
    this.rumble.gain.setTargetAtTime(0.18 + 0.4 * a.storm * (0.6 + 0.4 * swell), t, k);
    this.rumbleF.frequency.setTargetAtTime(180 + 120 * a.storm, t, k);
    this.wash.gain.setTargetAtTime(0.03 + 0.09 * a.storm * swell + 0.04 * a.heave, t, k);
    this.washF.frequency.setTargetAtTime(500 + 500 * swell, t, k);
    this.hull.gain.setTargetAtTime(0.02 + a.speed * 0.2 + Math.min(a.heave, 3) * 0.05, t, 0.1);
    this.hullF.frequency.setTargetAtTime(700 + a.speed * 900, t, 0.1);
    this.rainG.gain.setTargetAtTime(a.rain * 0.16, t, 0.5);
    this.windWalk = Math.min(1, Math.max(0, this.windWalk + (Math.random() - 0.5) * dt * 0.8));
    this.gustWalk = Math.min(1, Math.max(0, this.gustWalk + (Math.random() - 0.5) * dt * 1.2));
    this.wind.gain.setTargetAtTime((0.03 + 0.16 * a.wind) * (0.5 + this.windWalk), t, k);
    this.windF.frequency.setTargetAtTime(300 + 600 * this.windWalk + 300 * a.wind, t, k);
    this.gust.gain.setTargetAtTime(0.06 * a.wind * a.wind * this.gustWalk, t, k);
    this.gustF.frequency.setTargetAtTime(1300 + 1200 * this.gustWalk, t, k);
    const rpm = Math.abs(a.rpm);
    this.eng.frequency.setTargetAtTime(26 + rpm * 26, t, 0.3);
    this.engSub.frequency.setTargetAtTime(13 + rpm * 13, t, 0.3);
    this.engF.frequency.setTargetAtTime(120 + rpm * 160, t, 0.3);
    this.engG.gain.setTargetAtTime(0.05 + rpm * 0.1, t, 0.3);
  }
}
