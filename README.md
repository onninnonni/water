# Water — real-time Navier–Stokes in WebGL2

A GPU "stable fluids" solver (Stam) written in TypeScript + WebGL2, rendered to look like water.

- Drag with the mouse or finger to push the water.
- Reflecting (solid, free-slip) walls.
- Damping buttons: None / Low / Medium / High.
- Viscosity slider (implicit Jacobi diffusion).

Per frame: self-advect velocity → viscous diffusion → boundary → divergence → pressure Poisson (Jacobi) → gradient subtraction → boundary → advect surface field.

```bash
npm install
npm run dev     # local dev server
npm run build   # static bundle in dist/
```

Live: https://pikos.org/onni/water/
