// Engine and sea sound. Built lazily on the first user gesture; cheap procedural sources.
export class SeaAudio {
  private ctx: AudioContext | null = null;
  private engine!: OscillatorNode;
  private engine2!: OscillatorNode;
  private engineGain!: GainNode;
  private engineFilter!: BiquadFilterNode;
  private seaGain!: GainNode;
  private seaFilter!: BiquadFilterNode;
  private master!: GainNode;
  enabled = false;

  private build() {
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.master = ctx.createGain(); this.master.gain.value = 0; this.master.connect(ctx.destination);
    // Engine: two detuned saws through a lowpass.
    this.engineFilter = ctx.createBiquadFilter(); this.engineFilter.type = 'lowpass'; this.engineFilter.frequency.value = 400; this.engineFilter.Q.value = 2;
    this.engineGain = ctx.createGain(); this.engineGain.gain.value = 0;
    this.engine = ctx.createOscillator(); this.engine.type = 'sawtooth'; this.engine.frequency.value = 50;
    this.engine2 = ctx.createOscillator(); this.engine2.type = 'square'; this.engine2.frequency.value = 25.3;
    const g2 = ctx.createGain(); g2.gain.value = 0.35;
    this.engine.connect(this.engineFilter); this.engine2.connect(g2); g2.connect(this.engineFilter);
    this.engineFilter.connect(this.engineGain); this.engineGain.connect(this.master);
    this.engine.start(); this.engine2.start();
    // Sea: filtered noise.
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const d = buf.getChannelData(0);
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < len; i++) { // pink-ish noise
      const w = Math.random() * 2 - 1;
      b0 = 0.99765 * b0 + w * 0.099; b1 = 0.963 * b1 + w * 0.2965; b2 = 0.57 * b2 + w * 1.0526;
      d[i] = (b0 + b1 + b2 + w * 0.1848) * 0.11;
    }
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    this.seaFilter = ctx.createBiquadFilter(); this.seaFilter.type = 'lowpass'; this.seaFilter.frequency.value = 600;
    this.seaGain = ctx.createGain(); this.seaGain.gain.value = 0.15;
    src.connect(this.seaFilter); this.seaFilter.connect(this.seaGain); this.seaGain.connect(this.master);
    src.start();
  }

  toggle(): boolean {
    if (!this.ctx) this.build();
    this.enabled = !this.enabled;
    const ctx = this.ctx!;
    if (this.enabled && ctx.state === 'suspended') void ctx.resume();
    this.master.gain.setTargetAtTime(this.enabled ? 0.6 : 0, ctx.currentTime, 0.1);
    return this.enabled;
  }

  // throttle 0..1, speed 0..1 (normalised), slam 0..1 (hull impact this frame)
  update(throttle: number, speed: number, slam: number, driving: boolean) {
    if (!this.ctx || !this.enabled) return;
    const t = this.ctx.currentTime;
    const rpm = driving ? 0.25 + Math.abs(throttle) * 0.5 + speed * 0.45 : 0;
    this.engine.frequency.setTargetAtTime(38 + rpm * 70, t, 0.15);
    this.engine2.frequency.setTargetAtTime(19 + rpm * 35, t, 0.15);
    this.engineFilter.frequency.setTargetAtTime(250 + rpm * 900, t, 0.15);
    this.engineGain.gain.setTargetAtTime(driving ? 0.12 + rpm * 0.25 : 0, t, 0.2);
    this.seaFilter.frequency.setTargetAtTime(500 + speed * 2500 + slam * 3000, t, 0.08);
    this.seaGain.gain.setTargetAtTime(0.12 + speed * 0.35 + slam * 0.6, t, 0.08);
  }
}
