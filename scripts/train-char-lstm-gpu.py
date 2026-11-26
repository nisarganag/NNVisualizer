#!/usr/bin/env python3
"""
Trains the Recurrent page's networks on a Mac's GPU with MLX: the same models and recipe
as scripts/train-char-lstm.mjs, which runs on the CPU, many times faster.

Usage (Python 3.13 with mlx and numpy, for example in a uv virtual environment):
  node scripts/train-char-lstm.mjs <corpus-dir> <work-dir> --language hi --corpus-only
  python scripts/train-char-lstm-gpu.py <work-dir> --cell lstm [--epochs 8] [--seed N]
  node scripts/train-char-lstm.mjs <corpus-dir> public/models --language hi --cell lstm \\
       --import <work-dir>/gpu-lstm.bin

The first step writes the normalized training and held-out text and the vocabulary.
This script trains on them and writes <work-dir>/gpu-<cell>.bin (the best epoch's
parameters as little-endian float32, in the layout of src/rnn/model.ts) and
gpu-<cell>-history.json. The last step reads those back with the same code the page
runs, checks the held-out score, finds the notable cells and reference statistics,
and exports the model, exactly as after training on the CPU.

The recipe is train-char-lstm.mjs's: one-hot characters into one LSTM layer of 256
cells (gates input, forget, cell candidate, output) or a gateless tanh network of 512
units, and a softmax readout; truncated backpropagation through time over 100 steps
on 40 streams that carry their memory from one window to the next and start again from
zero with probability 0.1 (and at every epoch); the gradient of the mean loss clipped
at norm 1; Adam (0.9, 0.999, 1e-8) with the learning rate warming up over 150 updates
to 2e-3 (1e-3 for the gateless network) and cosine-decaying to 5e-5; the best epoch by
held-out bits per character. The initialization is drawn the same way (orthogonal
recurrent weights, forget bias 1, output bias at the log character frequencies) from
NumPy's generator rather than the CPU trainer's, and the GPU computes in float32 in a
different order, so the weights differ from a CPU run's; the scores agree closely.
"""
import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

STEPS = 100
STREAMS = 40
RESET_PROBABILITY = 0.1
FINAL_LEARNING_RATE = 5e-5
WARMUP_STEPS = 150
CLIP_NORM = 1.0
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8
CELLS = {
    'lstm': {'label': 'LSTM', 'hidden': 256, 'learning_rate': 2e-3, 'gates': 4},
    'rnn': {'label': 'Gateless RNN', 'hidden': 512, 'learning_rate': 1e-3, 'gates': 1},
}
LAYOUT = ['inputWeights', 'recurrentWeights', 'bias', 'outputWeights', 'outputBias']


def load(work):
    vocabulary = json.loads((work / 'corpus-vocabulary.json').read_text(encoding='utf-8'))
    lookup = {c: i for i, c in enumerate(vocabulary)}
    encode = lambda text: np.fromiter((lookup[c] for c in text), dtype=np.int32, count=len(text))
    train = encode((work / 'corpus-train.txt').read_text(encoding='utf-8'))
    held_out = [encode(book) for book in (work / 'corpus-held-out.txt').read_text(encoding='utf-8').split('\n\n')]
    return vocabulary, train, held_out


def orthogonal(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def initialize(cell, V, H, train, rng):
    gates = CELLS[cell]['gates']
    params = {'inputWeights': 0.3 * rng.standard_normal((V, gates * H))}
    if cell == 'lstm':
        params['recurrentWeights'] = np.concatenate([orthogonal(H, rng) for _ in range(4)], axis=0)
        bias = np.zeros(4 * H)
        bias[H:2 * H] = 1  # forget gate
    else:
        params['recurrentWeights'] = orthogonal(H, rng)
        bias = np.zeros(H)
    params['bias'] = bias
    params['outputWeights'] = rng.standard_normal((V, H)) / math.sqrt(H)
    frequency = np.bincount(train, minlength=V) + 1.0
    params['outputBias'] = np.log(frequency / (len(train) + V))
    return {name: mx.array(value.astype(np.float32)) for name, value in params.items()}


def step_lstm(p, x_in, h, c):
    H = h.shape[1]
    g = x_in + h @ p['recurrentWeights'].T + p['bias']
    i, f = mx.sigmoid(g[:, :H]), mx.sigmoid(g[:, H:2 * H])
    candidate, o = mx.tanh(g[:, 2 * H:3 * H]), mx.sigmoid(g[:, 3 * H:])
    c = f * c + i * candidate
    return o * mx.tanh(c), c


def step_rnn(p, x_in, h, c):
    return mx.tanh(x_in + h @ p['recurrentWeights'].T + p['bias']), c


def make_window(cell):
    step = step_lstm if cell == 'lstm' else step_rnn

    def window(p, x, y, mask, h, c):
        """Summed cross-entropy (nats) of a window, and the memory it ends with."""
        inputs = p['inputWeights'][x]
        outputs = []
        for t in range(x.shape[1]):
            h, c = step(p, inputs[:, t], h, c)
            outputs.append(h)
        logits = mx.stack(outputs, axis=1) @ p['outputWeights'].T + p['outputBias']
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, y[..., None], axis=-1)[..., 0]
        return (nll * mask).sum(), h, c

    return window


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('work', type=Path)
    parser.add_argument('--cell', choices=CELLS, default='lstm')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261119)
    parser.add_argument('--max-windows', type=int, default=0, help='stop each epoch early (for a quick check)')
    args = parser.parse_args()
    config = CELLS[args.cell]
    rng = np.random.default_rng(args.seed)
    vocabulary, train, held_out = load(args.work)
    V, H, N = len(vocabulary), config['hidden'], len(train)
    params = initialize(args.cell, V, H, train, rng)
    count = sum(int(np.prod(value.shape)) for value in params.values())
    window = make_window(args.cell)

    def loss_and_memory(p, x, y, mask, h, c):
        loss, h, c = window(p, x, y, mask, h, c)
        return loss, (h, c)

    # The loss is differentiated; the memory the window ends with rides along.
    gradient = mx.compile(mx.value_and_grad(loss_and_memory))
    forward = mx.compile(lambda p, x, y, mask, h, c: window(p, x, y, mask, h, c))

    def evaluate(p):
        """Bits per character on each book's held-out text, read from a blank memory, as one batch."""
        B, L = len(held_out), max(len(book) for book in held_out)
        x = np.zeros((B, L), np.int32)
        valid = np.zeros((B, L), np.float32)
        for b, book in enumerate(held_out):
            x[b, :len(book)] = book
            valid[b, :len(book) - 1] = 1  # every character after the first is predicted
        h = mx.zeros((B, H))
        c = mx.zeros((B, H))
        nats = 0.0
        for start in range(0, L - 1, STEPS):
            end = min(start + STEPS, L - 1)
            pad = STEPS - (end - start)
            xs, ys = x[:, start:end], x[:, start + 1:end + 1]
            ms = valid[:, start:end]
            if pad:
                xs, ys, ms = (np.pad(a, ((0, 0), (0, pad))) for a in (xs, ys, ms))
            loss, h, c = forward(p, mx.array(xs), mx.array(ys), mx.array(ms), h, c)
            mx.eval(loss, h, c)
            nats += loss.item()
        return float(nats / valid.sum() / math.log(2))

    windows_per_epoch = ((N - STEPS) // STREAMS - 1) // STEPS
    total_steps = args.epochs * windows_per_epoch

    def learning_rate(step):
        return min(1, step / WARMUP_STEPS) * (FINAL_LEARNING_RATE + 0.5 * (config['learning_rate'] - FINAL_LEARNING_RATE)
                                              * (1 + math.cos(math.pi * min(1, step / total_steps))))

    adam_m = {name: mx.zeros_like(value) for name, value in params.items()}
    adam_v = {name: mx.zeros_like(value) for name, value in params.items()}
    print(f"{config['label']} {H} {'cells' if args.cell == 'lstm' else 'units'}, {count:,} parameters; {N:,} training characters, "
          f"{args.epochs} epochs of about {windows_per_epoch} updates ({STREAMS} streams x {STEPS} steps) on {mx.default_device()}, seed {args.seed}", flush=True)
    start_time = time.time()
    step = 0
    best = {'bits': math.inf, 'epoch': 0, 'params': None}
    history = []
    ones = mx.ones((STREAMS, STEPS))
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        offset = int(rng.integers(STEPS))
        span = (N - 1 - offset) // STREAMS
        windows = (span - 1) // STEPS
        if args.max_windows:
            windows = min(windows, args.max_windows)
        starts = offset + np.arange(STREAMS) * span
        h = mx.zeros((STREAMS, H))
        c = mx.zeros((STREAMS, H))
        loss_sum = 0.0
        norm_sum = 0.0
        clipped = 0
        for w in range(windows):
            positions = starts[:, None] + w * STEPS + np.arange(STEPS)[None, :]
            reset = np.ones(STREAMS, bool) if w == 0 else rng.random(STREAMS) < RESET_PROBABILITY
            keep = mx.array((~reset).astype(np.float32))[:, None]
            h, c = h * keep, c * keep
            x, y = mx.array(train[positions]), mx.array(train[positions + 1])
            (loss, (h, c)), grads = gradient(params, x, y, ones, h, c)
            norm = mx.sqrt(sum((g * g).sum() for g in grads.values())) / (STREAMS * STEPS)
            mx.eval(loss, norm, h, c)
            norm_value = norm.item()
            loss_sum += loss.item()
            norm_sum += norm_value
            scale = (CLIP_NORM / norm_value if norm_value > CLIP_NORM else 1.0) / (STREAMS * STEPS)
            clipped += int(norm_value > CLIP_NORM)
            step += 1
            rate = learning_rate(step)
            correction1, correction2 = 1 - BETA1 ** step, 1 - BETA2 ** step
            for name in params:
                g = grads[name] * scale
                adam_m[name] = BETA1 * adam_m[name] + (1 - BETA1) * g
                adam_v[name] = BETA2 * adam_v[name] + (1 - BETA2) * g * g
                params[name] = params[name] - rate * (adam_m[name] / correction1) / (mx.sqrt(adam_v[name] / correction2) + EPSILON)
            mx.eval(params, adam_m, adam_v)
            if w and w % max(1, windows // 10) == 0:
                elapsed = time.time() - epoch_start
                print(f"  epoch {epoch}: {w}/{windows} updates, train {loss_sum / (w * STREAMS * STEPS) / math.log(2):.3f} bits/char, "
                      f"grad norm {norm_sum / w:.3f} (clipped {clipped}), lr {rate:.2e}, {w * STREAMS * STEPS / elapsed:,.0f} characters/s", flush=True)
        train_bits = float(loss_sum / (windows * STREAMS * STEPS) / math.log(2))
        validation_bits = evaluate(params)
        history.append({'epoch': epoch, 'trainBits': round(train_bits, 4), 'validationBits': round(validation_bits, 4)})
        if validation_bits < best['bits']:
            best = {'bits': validation_bits, 'epoch': epoch, 'params': {name: np.array(value) for name, value in params.items()}}
        print(f"epoch {epoch:2d}/{args.epochs}  lr {learning_rate(step):.2e}  train {train_bits:.3f} bits/char  held-out {validation_bits:.3f} bits/char  "
              f"grad norm {norm_sum / windows:.3f} (clipped {clipped}/{windows})  {time.time() - epoch_start:.1f} s", flush=True)

    seconds = time.time() - start_time
    weights = np.concatenate([best['params'][name].astype('<f4').ravel() for name in LAYOUT])
    (args.work / f'gpu-{args.cell}.bin').write_bytes(weights.tobytes())
    record = {'cell': args.cell, 'hidden': H, 'epochs': args.epochs, 'bestEpoch': best['epoch'], 'seed': args.seed,
              'seconds': round(seconds), 'device': f'{mx.default_device()} (MLX {mx.__version__})', 'history': history}
    (args.work / f'gpu-{args.cell}-history.json').write_text(json.dumps(record, indent=1))
    print(f"Best epoch {best['epoch']}: {best['bits']:.4f} bits/char held out. Wrote {args.work / f'gpu-{args.cell}.bin'} "
          f"({weights.nbytes:,} bytes) in {seconds:.0f} s.", flush=True)


if __name__ == '__main__':
    main()
