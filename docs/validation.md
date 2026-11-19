# Validation record

Checked on 2026-11-19.

## Automated checks

The final `npm run check` passes with **11 unit/reference/state tests and 16 Chromium browser tests**, plus TypeScript checking and the Vite production build. The complete redesigned application was verified on Node.js 24.22.3 (matching the recommended Vercel major version) on macOS arm64. Development and earlier browser checks also ran on Node.js 26.13.1.

- The reference fixture compares training/test losses and selected parameters at epochs 0, 1, and 10 against values independently captured from the live TensorFlow Playground, to 12 decimal places.
- Engine checks cover a hand-calculated gradient update, partial batches across epochs, parameter edits, data replacement, zero-input models, and numerical divergence.
- URL checks cover supported settings, encoded seeds, empty networks, malformed values, bounded network sizes, and safe lesson/control names.
- Browser checks cover every dataset, activation and regularization option; training and reset; loss history; feature and layer editing; weight and bias editing; test points and discrete heatmaps; URL restoration; classroom controls; lessons; divergence recovery; and small screens.
- Production requests include a real module worker and locally served fonts/icons. The default experiment loads without external requests or failed asset responses.
- A resize regression verifies that accessible names, keyboard feature toggles, disabled neuron limits, and hidden classroom controls survive redraws and resets.
- Studio navigation, including the wordmark, preserves the current configuration and epoch. Copy-link sharing uses the current URL. Classroom links remain synchronized and open the configured view separately.
- A mobile browser test uses actual touchscreen taps on visible SVG bias/weight targets, edits both values, and verifies that the worker applies them without advancing the epoch.

## Visual and independent review

The original reference-like interface was replaced with the studio design requested by the user. The development and production builds were checked in Chromium without console errors. Layouts were inspected at 1440 × 1000, 1024 × 900 (six hidden layers), and 390 × 844. The network scrolls horizontally on small screens while the document remains within the viewport. `docs/studio-preview.png` is a screenshot of the working production build after training the default experiment.

The initial independent review found a classroom visibility issue during graph redraws; that and the related accessibility state loss were fixed and regression-tested. The redesign received a separate independent review. Its touch-editor, wordmark-navigation, and classroom-link findings were reproduced with failing browser tests, fixed, and verified by the complete passing suite. Lesson color terminology, legend scaling, and the hidden-guide footer were also corrected. No identified issue was left deferred.

Independent browser checks additionally covered denied clipboard access, a bias-only model with no visible inputs, all 49 bias hit targets in a six-layer network, and document/control bounds at eleven widths from 320px through 1440px. Earlier checks confirmed eight queued clicks produce eight epochs and rapid reset sequences reject stale snapshots.

## Scope

The production build is configured for Vercel static hosting. Publishing to a Vercel account is a separate user step described in the README. Automated browser coverage is Chromium; no cross-browser certification is implied. Shared links restore experiment settings and the seed, not trained weights or the current epoch.
