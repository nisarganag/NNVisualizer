# Validation record

Checked on 2026-11-19.

## Automated checks

The final `npm run check` passes with **32 unit, reference, model and state tests and 23 Chromium browser tests**, plus TypeScript checking and the Vite production build, on Node.js 26.13.1 (macOS arm64). The previous studio version was also verified on Node.js 24.22.3, matching the recommended Vercel major version; the theme and motion changes add no Node-specific code and were not re-run on 24.

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

## Themes and motion

The light/dark redesign was checked the same day against the production build.

- Signal colors were validated per theme with an OKLab color-vision check (protanopia and deuteranopia simulation). Violet/amber separation is ΔE 31.9 (light) and 27.4 (dark), against a target of at least 8. Both poles clear 3:1 contrast on their surfaces: light 5.2–5.5 (violet) and 3.3–3.6 (amber); dark 6.7–7.5 and 8.0–8.9. The dark poles sit above the categorical lightness band by design, so hairline connections and small points stay visible on the near-black stage.
- Browser tests cover following the system theme, switching without resetting the experiment or URL, repainting the heatmap, thumbnails and connections, remembering the choice after reload, and reduced motion (no running animations after a structural edit or a theme switch). The neuron-inspection test also checks connection highlighting and typeset input names.
- Layouts were inspected in both themes at 1440 × 1000 and 390 × 844, plus mid-animation frames of the load sequence, a layer addition, and the theme reveal. Twelve widths from 320px to 1920px produce no horizontal page overflow.
- Production loads without a flash of unstyled controls, and new connections render at their final width without load-time transitions.

## Digits and Transformer

Added the same day and checked against the production build.

- **CNN (Digits):** 99.39% accuracy on the 10,000 MNIST test images (61 errors), using the shipped weights read back through the page's own inference code. On 1,404 synthetic brush-stroke digits rendered through the full drawing pipeline it was 97.4% correct, and 99.1% on 230 harder variants. Retraining reproduces byte-identical weights. Unit tests cover parsing, the shipped examples, activation shapes, preprocessing (blank input, centering) and synthetic strokes. A browser test draws a 1 and a 0 with the mouse, inspects feature maps, clears, and loads an example; another runs it by touch on a phone.
- **GPT-2 (Transformer):** the tokenizer matches the reference ids. Float32 inference reproduces the original model's greedy continuation of "The dog" token for token. The shipped 8-bit weights reproduce the same 18 tokens, and next-token KL divergence from float32 is 0.0004–0.0007 on the checked prompts (probabilities within about 0.3 percentage points). Plain per-row 8-bit quantization was measured at 0.0132 and rejected. A browser test loads the model from the production build, checks that the top five candidates for "The capital of France is" match the float32 model in order, exercises blocks, heads, temperature, top-p and generation, and confirms that a reload opens the cached model without asking.
- **Site:** all three pages were checked for horizontal overflow and header collisions at thirteen widths from 320px to 1920px, in both themes. Navigating between models preserves the playground experiment.

## Scope

The production build is configured for Vercel static hosting. Publishing to a Vercel account is a separate user step described in the README. Automated browser coverage is Chromium; no cross-browser certification is implied. Shared links restore experiment settings and the seed, not trained weights or the current epoch.
