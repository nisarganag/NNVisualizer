# Validation record

Checked on 2026-11-19.

## Automated checks

The final `npm run check` passes with **94 unit, reference, model and state tests and 50 Chromium browser tests** (46 in the headless shell, 4 in full Chromium on the GPU), plus TypeScript checking and the Vite production build, on Node.js 26.13.1 (macOS arm64). The previous studio version was also verified on Node.js 24.22.3, matching the recommended Vercel major version; the theme and motion changes add no Node-specific code and were not re-run on 24.

- The reference fixture compares training/test losses and selected parameters at epochs 0, 1, and 10 against values independently captured from the live TensorFlow Playground, to 12 decimal places.
- Engine checks cover a hand-calculated gradient update, partial batches across epochs, hand-drawn sets smaller than one batch, parameter edits, data replacement, zero-input models, and numerical divergence.
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

## Switching off heads, the gateless network and the autoencoder

Added the next day, each built separately and reviewed before merging.

- **Switching off heads (Transformer):** a switched-off head's output is replaced by zeros before the output projection, at every position, on both the CPU and the GPU; its attention pattern is still computed and shown. With nothing switched off, 24 of 24 runs on each processor are bit-for-bit identical to the code before the change. Under switching, the GPU matches the CPU in 48 comparisons with four masks, including every head off: the same top five, logits within 2.1e-4, the same lens, and surprisal within 4.4e-5 bits. Unit tests cover restoring the exact baseline after switching heads back on, prefix reuse under a mask, and an invariant: with every head off, the last position's prediction doesn't depend on earlier tokens, bit for bit. One switched-off run of 14 tokens takes 24 ms on the GPU; testing all 144 heads takes about a second (3 s at 51 tokens). The negative name-mover effect described in the README was measured on the page and agrees with Wang et al. (2022).
- **Gateless network (Recurrent):** an Elman network with 512 units (329,281 parameters, against the LSTM's 346,433) trained on the same books and split, with the same seed handling and 8 epochs, at a peak learning rate of 1e-3, in 100 minutes. Held out, it scores 1.85 bits per character (LSTM 1.77) and its first guess is right 61.3% of the time (LSTM 62.7%). The difference that the brief expected up close isn't there: both draw half of a guess's pull from the last 4 characters and 90% from the last 14 or 15. It appears further back: a character 100 back pulls 10 times harder on the LSTM, and 150 back 112 times, because the gateless share falls by about 6% per character. The page says exactly that. The best quotation unit reaches r = −0.29 against the LSTM's −0.74. The LSTM's statistics were added to its metadata from the shipped weights, without retraining; its weights, gradient check and forward pass are unchanged. Unit tests cover the gateless forward pass, prefix reuse, influence against finite differences, word previews, held-out bits per character, and the stored statistics, including the LSTM's longer tail.
- **Autoencoder:** a variational autoencoder with a two-number latent space (469,268 parameters), trained on MNIST in 17 minutes with 4 workers. On the test set: ELBO −138.7 nats (reconstruction 131.6, KL 7.1); a five-nearest-neighbor vote on the map names 81.6% of test digits correctly (4, 5 and 9 overlap most, at 65–67%); and the Digits network reads 82.1% of the reconstructions correctly. A short run reproduces the same weights with one and three workers. Unit tests cover parsing, output range, the stored codes of the examples (within 1e-5), reconstruction, class separation, class centers the Digits network reads correctly, and smooth decoding along all 45 paths between class centers. Browser tests cover dragging, keys, tapping by touch on a phone, drawing, examples, morphing, the decoded grid and a theme switch.
- **Site:** the header now has five model tabs, shown as icons below 900px. All five pages were checked for horizontal overflow and header collisions at nineteen widths from 320px to 1920px. The whole browser suite passed three times in a row (150 of 150).

## Your own data, and Hindi

- **Hand-drawn data:** a set with fewer training points than one batch used to never reach a batch boundary, so the network never trained on it (a flat loss and a wrong boundary after 1,500 epochs on 16 drawn points). Such a set now trains as one batch. The reference's datasets always fill a batch, so the parity test is unchanged; a unit test covers the small set, and the carried partial batch is checked with exact values.
- **CSV upload and drawing (Playground):** unit tests read both sample files back point for point and cover headers, semicolons with decimal commas, tabs, quotes, blank lines, CRLF and a byte-order mark, every error message (wrong column count, a value that isn't a number, a label other than 1, -1 or 0, an empty file), rescaling an axis that leaves the plot, the evenly spread subset of a large file (both classes kept when the file is sorted by label) and a note when every point has one class. A browser test downloads the sample, uploads it, trains, uploads a broken file (the message names the line, the dataset is unchanged), clears, uploads a file that needs rescaling, then does the same for regression: its own sample (a value column), an upload, a step of training and a value that isn't a number. Another browser test draws regression points with five values, trains, erases one, and restores the rest from the share link; values are stored to 0.01 and clamped to -1 to 1, which a unit test checks.
- **Hindi (Recurrent):** nineteen public-domain Hindi books from Hindi Wikisource, each first published before 1931, fetched as proofread page wikitext (pages not yet proofread left out) and normalized like the English books: editors' introductions, standalone verse, headings and 107 paragraphs with caste, communal or gendered slurs dropped; 5,339,281 training characters, a 76-character vocabulary, the last 5% of each book held out. Both networks were trained on an Apple M4's GPU with scripts/train-char-lstm-gpu.py (MLX 0.32.5), which follows the CPU trainer's recipe; its first two epochs' held-out scores (2.456 and 2.277 bits) track the CPU trainer's run on the same text (2.471 and 2.284). LSTM: 8 epochs in 29 minutes, 2.0852 bits per character held out; first guess right 58.7%, within three 77.4%; half of a guess's pull from the last 4 characters and 90% from the last 29. Gateless network: 8 epochs in 19 minutes, 2.1823 bits; 57.1% and 76.3%; 4 and 11 characters. scripts/train-char-lstm.mjs --import read both back with src/rnn/model.ts and scored the held-out text exactly as the GPU trainer did (2.0852 and 2.1823), on a 30-update trial first (4.8352 both ways). The word-length cell is clear (r = 0.68 for the LSTM, 0.60 for the gateless network); no cell follows quotations (best r = 0.16 and 0.03), so the page shows its quotation views only when the LSTM's cell reaches 0.5, which the English one does (−0.74). A browser test switches to Hindi and back, checks the Hindi networks load, that every character of the Hindi texts is known, that a vowel sign is its own step, and that Generate writes Devanagari; unit tests cover the language choice, Devanagari letters and marks, the dotted-circle display and the nukta normalization.

## Scope

The production build is configured for Vercel static hosting. Publishing to a Vercel account is a separate user step described in the README. Automated browser coverage is Chromium; no cross-browser certification is implied. Shared links restore experiment settings and the seed, not trained weights or the current epoch.
