# NNVisualizer design

The user approved implementation of the previously discussed TensorFlow Playground adaptation in the existing NNVisualizer project folder and selected Vercel for hosting.

## Goal

Reproduce the reference playground's layout, mathematical behavior, datasets, feature controls, network editing, training controls, loss plots, heatmaps, parameter inspection/editing, URL sharing, and teaching visibility controls. Start with the supplied five-by-eight Gaussian experiment (seed 0.68936, Tanh, L1 0.01, learning rate 0.001, batch 13, seven features, 50% training).

## Approach

Adapt tensorflow/playground at commit 02469bd3751764b20486015d4202b792af5362a6 under Apache-2.0. Preserve its engine, dataset generation, D3/SVG/Canvas rendering, and desktop geometry. A React rewrite was considered but adds unnecessary parity risk. Use Vite and modern TypeScript tooling, bundle fonts locally, retain copyright/license notices, identify the adaptation as NNVisualizer, and remove the upstream analytics integration.

## Architecture

The browser generates seeded examples and the initial model. A Web Worker owns training and computes activation grids, returning transferable numeric buffers and model/loss snapshots. The main thread handles controls, URL state, and SVG/Canvas painting. Generation identifiers reject stale results after reset or editing. Preserve the original epoch/batch accumulation, fixed output activations, squared-error loss, and L1 zero-crossing behavior.

URL state restores the configuration and seed, not trained weights. Validate incoming numerical and shape settings to prevent invalid links from breaking or freezing the page. Keep zero-input and zero-hidden-layer networks usable. Provide a clear paused state when numerical training diverges. Preserve original tutorial hooks with safe local paths and a graceful missing-lesson fallback.

## Delivery

Work directly in the requested, initially empty project folder. Supply npm scripts, lockfile, Vercel static-build configuration, license/provenance, concise run/deploy instructions, meaningful engine/state tests, and browser interaction coverage. Desktop remains faithful; narrow screens can scroll the complete network without clipping controls. No backend or secrets are needed. Publishing is left to the user's Vercel workflow.

## Validation

Check fixed-seed training against the unmodified reference, gradient/update correctness, URL round trips and malformed parameters, every dataset/activation/regularizer, zero and maximum network shapes, parameter editing, play/pause/reset races, heatmaps/loss charts, local assets, desktop/mobile layouts, and production build behavior.
