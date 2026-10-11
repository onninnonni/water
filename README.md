# Seven Seas

A multiplayer pirate game in TypeScript and raw WebGL2. Every player sails the same sea: crew a
40 m ship with three masts, six cannons and a bow harpoon, or join someone else's. Sail out of a
sheltered bay, through a gate of sea stacks and under a rock arch, into a storm where the swell
reaches 20 m, sea serpents hunt and the Leviathan waits.

Live: https://onnin.no (also https://pikos.org/onni/water/) · The original 2D fluid simulator: https://pikos.org/juha/water/

## Play

- **Start:** choose a name. The lobby lists every ship at sea with its crew; *Join crew* puts you on
  board, or *Sail my own ship* in the sheltered bay or the heart of the storm. Invite links
  (`?ship=ID`) highlight a ship in the list.
- **On deck:** `WASD` walk, mouse to look (click to capture it), `Shift` run, `C` first/third person,
  `Q` wave. Stand at a cannon and click to fire it; anywhere else a click swings your cutlass.
- **The wheel:** `E` on the quarterdeck. `W`/`S` set or furl sail, `A`/`D` steer, `C` overview.
- **Masts:** `E` at the foot of a mast, `W`/`S` to climb to the top, `E` to slide down.
- **The harpoon:** `E` at the bow, aim with the mouse, click to fire. It bites serpents, the
  Leviathan and other ships and ties them to you. Hold `R` or right mouse to reel in; click again to
  rip it out, which does more damage the longer it held.
- **Ship against ship:** cannons and harpoons hit other ships. With another ship alongside, `E` at
  the main-deck rail lays a gangplank that lashes the hulls together; `E` at its end to cross. The
  crews fight with cutlasses; the fallen stay down until it is settled. When the other crew is all
  down, whoever holds your wheel can press `G` to cut her loose.
- **The Leviathan:** a giant serpent that rises in the biggest seas, far out in the storm (see the
  red ring on the map), with its own war-drum and deep-choir music.
- **Touch:** left thumb moves or steers, right thumb looks, buttons for everything else.

## How it works

- **Sea:** 12 deep-water Gerstner waves (ω = √(g·k)), 9 m to 300 m long, shared by the vertex shader
  and the CPU so the ship floats on exactly what you see. Calm near the island, 20 m seas 3 km out.
- **Ship:** buoyancy springs on heave, pitch and roll; carried by orbital currents and downwind
  drift; slides down wave faces; sails drive best with the wind behind; slams, green water, rocks.
- **Serpents:** host-simulated heads with bodies laid along their trail, arching through the swell;
  they circle, rear and ram. Cannonballs skip off the water and the guns time the roll.
- **Shared world:** everyone joins one relay room (up to 32 players). Each ship is simulated by its
  owner, who broadcasts snapshots; the first player in the world runs the serpents and the clock.
  Ownership passes to a crew member if the owner leaves. Water is masked inside every hull so big
  waves never wash through the deck. Relay: a small WebSocket server on pikos.org.
- **Sound:** procedural Web Audio: dark pads, deep horn-choir moans in the storm, irregular hull
  knocks and timber groans, wind, rain, cannons, serpent roars and distance-delayed thunder.

```bash
npm install
npm run dev     # local dev server
./deploy.sh     # build and upload to pikos.org
```
