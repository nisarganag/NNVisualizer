# Neural Network Visualizer themes, palette and motion

The user asked for a cooler, more aesthetic design with smoother animation, light and dark modes, and a good color palette. They described outcomes rather than methods, so the visual direction below is the implementer's choice. Function, numerical behavior, URLs, and the tested DOM contract are unchanged.

## Direction

**Color only means signal.** Interface chrome (buttons, text, borders, focus rings, charts) is neutral ink. Only data wears color: violet for positive values, amber for negative ones, with a neutral midpoint close to the chart surface. The earlier mint UI accent also meant "positive", so the interface and the data competed; now they don't.

**One family, two voices.** Mona Sans (variable width and weight) carries the interface and expanded headlines. Its tabular figures, with a slashed zero and footed one, carry live readouts, so epoch counts and losses keep their width while they change. Hubot Sans was tried for readouts and rejected because its barred zero reads as θ, which collides with parameter notation. Input features are typeset as math: italic variables, upright functions, and true sub/superscripts.

**Motion follows cause and effect.** The network wires itself up once on load. Structural edits glide surviving neurons to their new places and fade the rewired connections in. Signals flow along connections while training, as continuous CSS motion independent of the epoch rate. Hovering a neuron highlights its connections. The theme change is revealed as a circle growing from the toggle. Redraws that keep the structure (resizes, sliders, theme) do not animate. Reduced motion disables all of it.

## Palette

| Token | Light | Dark |
|---|---|---|
| Page | `#f3f4f9` | `#0f121e` |
| Surface | `#ffffff` | `#171a27` |
| Stage | `#f9f9fc` | `#0b0d1a` |
| Ink | `#1c1e2d` | `#eff0f6` |
| Positive (violet) | `#744ed7` | `#a692fb` |
| Negative (amber) | `#cf6d01` | `#f09e3e` |
| Neutral midpoint | `#eaebf0` | `#292b35` |

Each theme has its own steps (derived in OKLCH), not an inversion. Both pairs pass color-vision-deficiency separation, normal-vision separation, and 3:1 contrast. The dark poles deliberately exceed the categorical lightness band so thin marks stay luminous. Heatmap ramps interpolate in Lab. CSS custom properties are the single source; canvas and SVG rendering read them through `src/palette.ts` and repaint on `nnvisualizer:theme-change`.

## Theme behavior

An inline head script applies the stored or system theme before first paint. The toggle stores an explicit choice in `localStorage` (`nnvisualizer-theme`). Without one, the page follows system changes live. The theme never enters the URL hash, which belongs to experiment state.

## Constraints kept

No new runtime dependencies beyond the font. All assets are local. Existing IDs, classes, and accessible names are preserved. The neuron stepper's DOM order now matches its visual order (remove, then add). The loss-chart test asserts drawn segments rather than straight `L` commands, because the curves are now monotone.
