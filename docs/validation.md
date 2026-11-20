# Validation record

Checked on 2026-11-19.

## Automated checks

The final `npm run check` passes with **63 unit, reference, model and state tests and 37 Chromium browser tests** (35 in the headless shell, 2 in full Chromium on the GPU), plus TypeScript checking and the Vite production build, on Node.js 26.13.1 (macOS arm64). The previous studio version was also verified on Node.js 24.22.3, matching the recommended Vercel major version; the theme and motion changes add no Node-specific code and were not re-run on 24.

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

## Explainer additions

- **Playground:** SGD keeps the original update; the reference parity test still passes. Momentum and Adam are checked against hand-computed updates, GELU's derivative against finite differences, and hand-placed points against an encode/decode round trip with malformed input rejected. Browser tests cover every optimizer and new activation, drawing, erasing, sharing, reloading and clearing a dataset, and the backprop view's per-weight gradients and step replay.
- **Digits:** the input gradient matches central finite differences wherever the network is locally linear. The attack reached its target for all 108 sample and target pairs (median 18 pixel changes) and for 98.9% of 360 distorted samples, never changing a pixel twice.
- **Transformer:** the lens's last row equals the model's logits exactly; incremental surprisal equals a from-scratch computation; " Paris" after "The capital of France is" scores 4.8 bits against 21.2 for " banana". A browser test types a misspelt prompt and checks that the typo is marked most surprising.
- **Navigation:** pages no longer crossfade between models; Chrome aborted that fade with an unhandled error when clicks came in quick succession. Repeated runs of all model-page tests pass (32 of 32).

## Recurrent network, WebGPU and the prompt box

Added the same day.

- **Recurrent (LSTM):** retrained after feedback that its predictions and generated text made little sense. The first version (128 cells, four books, 40 epochs) is replaced by 256 cells trained for 8 epochs on nineteen books (5.1 million characters; 97 paragraphs with period slurs dropped, and *Tom Sawyer* left out for its language). On all held-out text (the last 5% of every book, 264,000 characters) it scores 1.77 bits per character (1.57 to 2.07 by book), measured on the shipped file with the page's own forward pass. On the held-out ends of the four original books, which neither version trained on:

  | | Earlier model | New model |
  |---|---|---|
  | Bits per character | 1.92 | 1.71 |
  | First guess right | 60.5% | 63.8% |
  | Right within three guesses | 80.3% | 83.1% |
  | Made-up words when generating (earlier default: temperature 0.8, any character) | 10.7% | 5.5% |
  | Made-up words with the new default (temperature 0.6, five likeliest) | 4.7% | 2.2% |

  A generated word counts as made up when it never occurs in the training text; each figure covers about 1,000 words from 24 samples. Sentences still don't make sense, and the page now says so. The trainer's gradient check passes; the earlier model was reproduced byte for byte from its seed, and this one, which took 105 minutes to train, was not retrained to check. Unit tests cover parsing and size checks, gate ranges and normalized probabilities, prefix reuse matching a fresh run bit for bit, the influence (gradient × input) against finite differences, a span limited to the latest characters matching the full one there, word previews matching a step-by-step run, held-out bits per character (under 1.65 on the test passage, which the earlier model fails at 1.68), and text encoding. Browser tests cover typing, choosing steps by pointer and keyboard, word previews and the five-candidate filter, the score, following a memory cell, switching the gate map, temperature, Generate and Stop, a theme switch, typography and unknown characters, reduced motion, and a 320px phone. On a 400-character text, choosing a step costs about 37 ms (30 ms for the memory span, now traced over the latest 200 characters, and 7 ms for the word previews) and each new character 0.3 ms.
- **WebGPU:** on an Apple M4 with Metal, the GPU path's logits are within 7.6e-5 of the CPU path's (about 4e-7 relative) for prompts of 5 to 145 tokens, including inputs split into several batches and a growing key/value cache. The top five tokens and every rank are identical, surprisal agrees within 6e-5 bits and the lens within 5e-6, and greedy decoding of "The dog" reproduces the 18 reference tokens. A browser test compares the GPU and the CPU on the same page (top five, logits within 1e-3, surprisal, lens) and generates greedily on the GPU. Losing the GPU, simulated by crashing Chromium's GPU process while idle, mid-run and mid-lens, moves the page to the CPU without errors.
- **Speed:** medians of warm runs on the same machine. "In the page" is the time the worker reports while the page draws its results, as visitors see it; "isolated" was measured without the page drawing.

  | | CPU | GPU, in the page | GPU, isolated | Software GPU (SwiftShader) |
  |---|---|---|---|---|
  | 9-token prompt | 746 ms | 27 ms | 10 ms | 310 ms |
  | 45-token prompt (48 isolated) | 3.7 s | 75 ms | 37 ms | 1.2 s |
  | One generated token | 87 ms | 10 ms | 3.8 ms | 340 ms |
  | Logit lens | 355 ms | 11 ms | 3.5 ms | 108 ms |

  A software GPU is used only when `?backend=gpu` asks for it, since it generates more slowly than the CPU path. The CI configuration (the Linux headless shell with software WebGPU) has not been run on Linux; if a runner offers no adapter, the GPU tests skip with a reason.
- **Prompt box:** the Transformer prompt is now a text area, so line breaks GPT-2 writes stay in it. A browser test generates two line breaks greedily, types after them and checks that the tokens equal GPT-2's own continuation; it fails with the previous single-line field. On both pages, generated text stays in view once the box reaches its maximum height.
- **Site:** all four pages were checked for horizontal overflow and header collisions at sixteen widths from 320px to 1920px, and the navigation test now visits every model. A playground test read connection widths during their 0.3 s transition and failed once under load; it now waits for them. The whole browser suite then passed four times in a row (148 of 148). The README screenshots were retaken for all four pages in both themes.

## Scope

The production build is configured for Vercel static hosting. Publishing to a Vercel account is a separate user step described in the README. Automated browser coverage is Chromium; no cross-browser certification is implied. Shared links restore experiment settings and the seed, not trained weights or the current epoch.
