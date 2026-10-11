# Seven Seas

A storm-sailing game in TypeScript and raw WebGL2. Take a 14 m trawler out of a sheltered bay,
through a gate of sea stacks and under a rock arch, into an open ocean where the swell builds to
10 m seas, the rain thickens and lightning walks the horizon.

Live: https://pikos.org/onni/water/

## Controls

- **Desktop:** `W`/`S` move the throttle lever (it stays where you leave it), `A`/`D` steer,
  `X` all stop, `C` switches between the bridge and a chase camera, drag to look around, `M` mutes.
- **Touch:** drag sideways on the left half to steer, drag up and down on the right half to set the throttle.

## How it works

- **Sea:** 12 deep-water Gerstner waves (dispersion ω = √(g·k)), 6 m to 220 m long. Long swell
  scales from almost nothing near the island to full storm 3 km out; short chop stays a little
  everywhere. The same wave function runs in the vertex shader and on the CPU, so the hull floats
  on exactly what you see.
- **Boat:** buoyancy springs on heave, pitch and roll from five hull samples; the hull is dragged by
  the water's orbital currents and a downwind drift, slides down wave faces, slams into crests,
  and collides with rocks.
- **World:** procedural crescent island with cliffs, beaches, six sea stacks and an arch, a lighthouse
  with a rotating beam, and channel and storm buoys. The ocean is a camera-centred polar grid out to 7 km.
- **Atmosphere:** storm intensity follows distance from the island and drives fog, cloud cover,
  low scud, rain, wind, whitecaps and lightning frequency.
- **Sound:** all procedural Web Audio: dark ambient pads and drone, sonar pings, ocean rumble, hull
  wash, wind, rain, diesel, bow slaps, creaks and distance-delayed thunder.

```bash
npm install
npm run dev     # local dev server
./deploy.sh     # build and upload to pikos.org
```
