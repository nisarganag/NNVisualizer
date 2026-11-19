# Digits (CNN) and Transformer pages

The user asked to grow Neural Network Visualizer beyond the neural-network playground: a Transformer explainer in the spirit of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/), and a CNN that classifies digits drawn on a scratchpad. After learning that Transformer Explainer runs the real GPT-2 small model in the browser, the user chose the same: the real model, live, with visible weights and layers. They approved building both.

## Site structure

Three pages share one header, theme system, palette and type: Playground (`/`, unchanged URLs and share links), Digits (`/cnn/`) and Transformer (`/transformer/`). The header's primary navigation becomes a model switcher. The playground's Learn and Classroom shortcuts move into the page. A multi-page Vite build keeps each page's code separate, and cross-page view transitions smooth navigation where supported.

## Digits (CNN)

- Scratchpad for mouse, touch and pen, classified live while drawing. Clear and Try an example (one MNIST test digit per class).
- The drawing is preprocessed like MNIST: ink bounding box scaled into 20×20, centered by center of mass in 28×28. The page shows the 28×28 input it actually sees.
- Network: Conv 8×3×3 → max pool → Conv 16×3×3 → max pool → Dense 64 → 10 outputs, about 52k parameters, trained here on MNIST with augmentation for hand-drawn input. Weights (≈200 KB) ship with the site and inference runs in the browser.
- Every layer's activations are drawn. Hovering a feature map shows the filter weights that produced it. Signed weights use the violet/amber diverging palette; non-negative activations use a single-hue ramp.

## Transformer (GPT-2 small)

- Real GPT-2 small (124M parameters, MIT license). The official 548 MB weights are converted once to 8-bit per-row quantized matrices (about 125 MB), split into files under 25 MB, and served from the site itself. No external requests.
- Inference is our own TypeScript implementation in a Web Worker, so every intermediate value is available: tokens, token and position embeddings, Q/K/V for every layer and head, masked scaled scores, softmax attention, MLP activations, and final logits. A key/value cache keeps typing and generation responsive.
- The model downloads when the visitor chooses to load it. The size is shown first, and the files are kept in the browser's Cache Storage, so later visits load from the cache.
- Views: prompt and tokens; embeddings; a 12-block overview; for the selected block and head, Q/K/V, the attention matrix with the causal mask, and the head's weight matrices; the MLP; and next-token probabilities with temperature, top-k and top-p. Generate samples and appends a token.
- Credits: GPT-2 (OpenAI, MIT), tokenizer files from the same release, and Transformer Explainer (Georgia Tech Polo Club, MIT) as the inspiration. No code or assets are copied from it.

## Constraints

Static Vercel hosting; no backend or keys. Themes, reduced motion, keyboard access and responsive layouts apply to the new pages. Existing playground behavior and tests are preserved. Conversion and training scripts are committed for reproducibility; downloaded data stays outside the repository.
