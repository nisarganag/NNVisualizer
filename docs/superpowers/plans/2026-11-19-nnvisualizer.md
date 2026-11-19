# NNVisualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a working TensorFlow Playground adaptation in the user's project folder, ready for static deployment to Vercel.

**Architecture:** Preserve the original numerical engine and visual renderer. Extract training and heatmap calculations into a browser worker, share typed messages with the UI, validate URL configuration, and produce a static Vite build.

**Tech Stack:** TypeScript, Vite, D3 v3, Canvas, SVG, seedrandom, Web Workers, Vitest, Playwright, Vercel.

**Spec:** `docs/superpowers/specs/2026-11-19-nnvisualizer-design.md`

## Global Constraints

- Work in the requested NNVisualizer folder; no external worktree or publication.
- Reference commit: 02469bd3751764b20486015d4202b792af5362a6; retain Apache-2.0 notices.
- Preserve all visible reference features and the supplied initial experiment.
- Node >=22.12; recommend Node 24 for Vercel; production output `dist/`.
- Training and all runtime assets are local to the browser/site; no backend or API keys.

## Review Focus

- Rapid resets or configuration changes while training must reject stale worker results.
- Zero selected inputs and zero hidden layers must still form a valid bias-only model.
- Invalid/nonfinite/oversized URL parameters must normalize to safe usable settings.
- High learning rates must pause visibly on divergence rather than freezing the interface.
- Production build must load workers, fonts, icons, and sharing URLs without external dependencies.

## Task 1: Reference engine and state contracts

**Files:** `src/nn.ts`, `src/dataset.ts`, `src/features.ts`, `src/state.ts`, `src/model-state.ts`, `src/simulation.ts`, `tests/*.test.ts`, build configs, license/provenance.

**Interfaces:** `TrainingEngine` accepts a typed network config, serializable model, and train/test examples; exposes `step(settings)`, `replaceData(train, test)`, `editParameter(kind, id, value)`, and `snapshot(density)`. `State.deserializeState(hash)` and `state.toHash()` normalize and round-trip configuration. Model serialization preserves IDs, weights, biases and L1 dead links.

- [x] Add tests for known forward/update results, partial batches, safe URL restoration, zero-sized inputs, editing, divergence, and fixed-seed reference results.
- [x] Run tests before the corresponding new behavior; confirm the expected failure.
- [x] Import the pinned reference engine/data, implement the typed simulation and state changes.
- [x] Run `npm test`; expected: all engine/state tests pass.

## Task 2: Complete browser application

**Files:** `index.html`, `styles.css`, `src/main.ts`, `src/playground.ts`, `src/heatmap.ts`, `src/linechart.ts`, `src/training.worker.ts`, `src/worker-protocol.ts`, `public/`, `tests/browser/playground.spec.ts`.

**Interfaces:** Worker requests carry a generation ID and init/step/data/edit command; responses carry losses, epoch, model state, and transferable grids. UI applies only the current generation and owns the existing D3 graph/control selectors.

- [x] Add browser tests covering default configuration, training controls, all datasets/settings, structural edits, live parameter editing, URL reload, teaching controls, zero-input networks, and reset races.
- [x] Preserve reference layout and adapt the controller to worker snapshots; retain all controls and explanatory content.
- [x] Bundle fonts/icons locally, add accessible control names, support narrow-screen scrolling, and pause clearly on worker/numerical errors.
- [x] Run `npm run build` and `npm run test:e2e`; expected: production assets build and browser tests pass.

## Task 3: Vercel delivery and review

**Files:** `README.md`, `vercel.json`, `.github/workflows/check.yml`, provenance and validation notes.

- [x] Document local development, production preview, complete features, attribution, and Vercel import settings.
- [x] Check production worker/font requests, loss/heatmap behavior, and desktop/mobile screenshots.
- [x] Run `npm run check`; expected: unit tests, typecheck, build, and browser suite pass.
- [x] Obtain one independent code review, address material findings with regression checks, and leave a local preview available.

## Execution record

- User's implementation request authorizes executing the agreed architecture; no additional plan approval needed.
- The initially empty requested folder is the workspace; existing diagnostic files are left intact and ignored.

- Completed the engine, worker/UI, and Vercel delivery tasks. Fresh checks passed: 11 unit/reference/state tests, 12 Chromium browser tests, typecheck, and production build.
- Preserved D3 v3 with a scoped ES-module compatibility transform for development and production.
- Final review identified and independently verified the classroom-control redraw fix; no findings remain.
- Validation details are recorded in `docs/validation.md`.
- Repeated the complete check under Node.js 24.22.3 to verify the documented Vercel runtime; all 23 tests and the production build passed.
