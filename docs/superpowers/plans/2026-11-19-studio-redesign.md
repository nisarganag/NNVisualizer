# NNVisualizer Studio Redesign Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation and one final independent review.

**Goal:** Deliver a distinct, polished visual design while retaining the original neural playground's functionality.

**Architecture:** Replace HTML/CSS and adapt the graph layout and color rendering. Preserve the numerical engine, worker protocol, state validation and reference fixtures. Keep navigation/sharing helpers separate from the simulation controller.

**Tech stack:** Existing TypeScript, Vite, D3, Canvas/SVG, Web Worker and Vercel; locally bundled Manrope and IBM Plex Mono.

**Spec:** `docs/superpowers/specs/2026-11-19-studio-redesign-design.md`

## Global constraints

- Work in the requested project folder; the user's redesign request authorizes these reversible UI changes.
- Numerical behavior, supported controls and default experiment stay the same.
- URL hash belongs to experiment state; in-page navigation must preserve it and training progress.
- Preserve license attribution and keep runtime assets local.
- Native controls and accessible names must work after reset and resize.

## Review focus

- Zero/six hidden layers, all features disabled and hidden classroom controls must remain usable.
- New graph geometry must preserve accurate weight/bias hit targets and hover previews.
- Narrow layouts must not clip settings or overflow the document.
- Guide/share actions must preserve current URL configuration and epoch.
- Fonts, chart colors and heatmap colors must agree in development and production.

## Tasks

1. **Studio shell and visual language** — `index.html`, `styles.css`, `src/main.ts`, `src/studio.ts`, `src/palette.ts`, fonts/favicon/notices. Replace the original page composition; add real navigation/share actions and a responsive layout.
2. **Graph and results presentation** — `src/playground.ts`, `src/heatmap.ts`, `src/linechart.ts`. Center circular nodes in a responsive canvas, add an output preview, restyle layer controls and parameter editing, use a shared palette, and display actual model/data metrics.
3. **Verify and deliver** — browser regressions, desktop/mobile visual inspection, numerical parity, production build, one independent review, updated README/validation notes, refreshed preview.

- [x] Add and observe navigation/sharing regression failures before implementing those behaviors.
- [x] Complete the studio shell and graph renderer updates.
- [x] Run the existing functional suite, fixing real regressions and updating changed presentation expectations.
- [x] Inspect desktop, intermediate and mobile layouts; verify six-layer and zero-layer configurations.
- [x] Obtain final review and resolve material findings.
- [x] Run `npm run check` under Node 24, update documentation, and show the redesigned preview.

## Execution record

- The latest user instruction replaces the previous instruction to reproduce the reference layout. The simulation remains the behavioral reference.
- Existing tests cover numerical behavior; visual-only CSS choices are checked through browser screenshots rather than implementation-mirroring tests.

- Completed the original studio shell, responsive graph/results renderer, native controls, local fonts/icons, and navigation/sharing helpers.
- Removed the previous Material Design and Roboto/Icon dependencies; updated complete third-party notices for the new fonts.
- All 27 final tests passed under Node 24.22.3, including unchanged numerical reference fixtures, desktop editing and real mobile touch editing. Typecheck and production build passed.
- Review fixes: pointer-transparent parameter previews; wordmark navigation without resetting progress; classroom links synchronized on state serialization and opened separately; correct mint/coral lesson text and a complete legend gradient; no guide link when explanatory text is hidden.
- Scope ruling: lesson color terminology and the clipped positive legend endpoint are learning/interpretation errors, so they were corrected with the other review findings rather than deferred as cosmetic changes.
- User authorized publication to https://github.com/nisarganag/NNVisualizer.git after completion. The remote was confirmed empty; publish on main without force-pushing.
