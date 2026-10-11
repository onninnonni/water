# Seven Seas

A two-player pirate game in TypeScript and raw WebGL2. Crew a 40 m ship with three masts, six
cannons and a bow harpoon. Sail out of a sheltered bay, through a gate of sea stacks and under a
rock arch, into a storm where the swell reaches 20 m and sea serpents hunt.

Live: https://pikos.org/onni/water/ · The original 2D fluid simulator: https://pikos.org/juha/water/

## Play

- **Start:** choose a name and spawn in the sheltered bay or the heart of the storm. *Sail solo*, or
  *Sail with a mate* to get an invite link; your mate opens it and joins your ship.
- **On deck:** `WASD` walk, mouse to look (click to capture it), `Shift` run, `C` first/third person,
  `Q` wave. Stand at a cannon and click or press `Space` to fire it.
- **The wheel:** walk to it on the quarterdeck and press `E`. `W`/`S` set or furl sail, `A`/`D` steer,
  `C` switches between your own view and an overview of the ship. Only one player holds the wheel;
  with nobody at it the ship drifts with the sea.
- **The harpoon:** press `E` at the bow ballista, aim with the mouse, click to fire.
- **Touch:** left thumb moves or steers, right thumb looks, buttons for the wheel, harpoon and cannons.

## How it works

- **Sea:** 12 deep-water Gerstner waves (ω = √(g·k)), 9 m to 300 m long, shared by the vertex shader
  and the CPU so the ship floats on exactly what you see. Calm near the island, 20 m seas 3 km out.
- **Ship:** buoyancy springs on heave, pitch and roll; carried by orbital currents and downwind
  drift; slides down wave faces; sails drive best with the wind behind; slams, green water, rocks.
- **Serpents:** host-simulated heads with bodies laid along their trail, arching through the swell;
  they circle, rear and ram. Cannonballs skip off the water and the guns time the roll.
- **Crew:** the first player is host and runs the simulation; others send inputs and avatar state.
  Messages go through a small WebSocket relay on pikos.org (`/onni/seas/ws/<room>`).
- **Sound:** procedural Web Audio: dark pads, deep horn-choir moans in the storm, irregular hull
  knocks and timber groans, wind, rain, cannons, serpent roars and distance-delayed thunder.

```bash
npm install
npm run dev     # local dev server
./deploy.sh     # build and upload to pikos.org
```
