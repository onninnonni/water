# Seven Seas

A multiplayer pirate game in TypeScript and raw WebGL2. Every player sails the same sea: crew a
40 m ship with three masts, six cannons and a bow harpoon, or join someone else's. Sail out of a
sheltered bay, through a gate of sea stacks and under a rock arch, into a storm where the swell
reaches 20 m, sea serpents hunt and the Leviathan waits.

Live: https://onnin.no · The original 2D fluid simulator: https://pikos.org/juha/water/

## Play

- **Start:** choose a name. The lobby lists every ship at sea with its crew; *Join crew* puts you on
  board, or *Sail my own ship* in the sheltered bay or the heart of the storm. Invite links
  (`?ship=ID`) highlight a ship in the list.
- **On deck:** `WASD` walk, mouse to look (click to capture it), `Shift` run, `C` first/third person,
  `Q` wave. Stand at a cannon and click to fire it; anywhere else a click swings your cutlass.
- **The wheel:** `E` on the quarterdeck. `W`/`S` set or furl sail, `A`/`D` steer, `C` overview.
- **Masts:** `E` at the foot of a mast, `W`/`S` to climb, `E` to slide down. Halfway up, `E` steps
  onto the railed platform, where you can walk around the mast (and fight); `E` again to climb on.
- **The harpoon:** `E` at the bow, aim with the mouse, click to fire. It bites serpents, the
  Leviathan and other ships and ties them to you. A harpooned serpent tows the ship by the bow,
  swinging her round so it is always ahead or abeam, never astern. Hold `R` or right mouse to reel in; click again to
  rip it out, which does more damage the longer it held. Reeling in on the Leviathan hauls the ship
  after it like a sled.
- **Ship against ship:** cannons and harpoons hit other ships. With another ship alongside, `E` at
  the main-deck rail lays a gangplank that lashes the hulls together; `E` at its end to cross. The
  crews fight with cutlasses; the fallen stay down until it is settled. When the other crew is all
  down, whoever holds your wheel can press `G` to cut her loose.
- **Wings:** folded wings hang at every masthead. From a mast platform or the top of a mast, `J`
  jumps off and glides: look down (or `W`) to dive and gain speed, up (`S`) to stretch the glide,
  `A`/`D` to turn. Land on any deck; the sea or the rocks mean being fished out and hauled aboard.
- **Riding the Leviathan:** land on its head, then set two hooks before it shakes you off: left
  (click or `Q`) and right (right-click or `E`). Hooked in, `A`/`D` steer it and `W`/`S` drive it
  faster or slower; steer it into an enemy ship to ram her. `J` leaps off.
- **The Leviathan:** a giant serpent that rises in the biggest seas, far out in the storm (see the
  red ring on the map), with its own war-drum and deep-choir music.
- **The storm:** in the heart of the storm the ship can fly off a crest and slam into the trough
  below: a splash-down throws white water over the deck, staggers the crew and drenches the lens.
- **Into the deep:** riding the Leviathan with both hooks set, `X` flips the hooks to bare the soft
  underside of its scales. It sounds at once, carrying you down to an air pocket in a cave where a
  ghost ship waits: half again as big, faster, with heavier, quicker cannons and a bow fire-thrower
  whose fireballs set the sea ablaze, burning any ship (or serpent) in it. Take her wheel and set
  sail to run the drowned tunnel and breach the surface where you went down. Your old ship stays
  with her crew, or sinks if you sailed her alone.
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
