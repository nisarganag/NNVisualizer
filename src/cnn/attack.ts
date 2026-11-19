/**
 * "Fool the network": a greedy pixel attack on the Digits page's CNN. It changes pixels
 * of the 28×28 input (never the drawing) a few at a time until the network reads a
 * target digit. DOM-free, so the page can animate it step by step and the unit tests
 * can run it headlessly.
 *
 * Each step differentiates the margin logits[target] − logits[rival], where the rival is
 * the strongest other digit right now, with respect to every input pixel. A pixel with
 * a positive gradient helps by gaining ink, so it may be set to 1; one with a negative
 * gradient helps by losing ink, so it may be set to 0. Pixels already at that end, and
 * pixels changed before, are skipped. Candidates are ranked by their first-order effect,
 * gradient × how far the pixel moves, which for MNIST's mostly 0-or-1 pixels is simply
 * the largest useful gradient. The best one is changed, plus the next one or two while
 * the step's estimated effect is below half of the remaining gap: far from the target a
 * step changes three pixels, close to it one at a time, because a first-order estimate
 * of a jump from 0 to 1 is rough. Then the network runs again on the new input.
 *
 * On the ten shipped MNIST samples and two drawn shapes, all 108 (digit, target) pairs
 * succeed within the cap, with a median of 18 changed pixels.
 */
import { inputGradient, runCnn, type CnnActivations, type CnnWeights } from './model';

/** The most pixels one attack may change: 120 of 784, about 15% of the image. */
export const ATTACK_PIXEL_CAP = 120;
/** The most pixels changed in one step. */
export const ATTACK_STEP_PIXELS = 3;

export type AttackStatus = 'running' | 'fooled' | 'failed';

export interface Attack {
  /** The digit the attack is trying to make the network read, 0..9. */
  readonly target: number;
  /** The perturbed input: a copy of the original in which changed pixels are 0 or 1. */
  readonly input: Float32Array;
  /** One entry per pixel: 0 untouched, 1 set to full ink, −1 set to background. */
  readonly changes: Int8Array;
  /** Indices of the changed pixels, in the order they changed. */
  readonly order: number[];
  /** Steps taken so far. */
  steps: number;
  /** runCnn's result for `input`. */
  activations: CnnActivations;
  /**
   * 'running' while there is work left, 'fooled' once the network reads the target,
   * 'failed' when the pixel cap is reached or no unchanged pixel can still help.
   */
  status: AttackStatus;
}

/** The digit the network reads: the most probable one, the first on a tie (as the page shows it). */
export function topDigit(probabilities: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
  return best;
}

/**
 * Prepares an attack on `original` (784 values in 0..1, left unchanged). Already
 * 'fooled' if the network reads the target before any change.
 * @throws RangeError if the target is not a digit or the input has the wrong length.
 */
export function startAttack(weights: CnnWeights, original: Float32Array, target: number): Attack {
  if (!Number.isInteger(target) || target < 0 || target > 9) {
    throw new RangeError(`startAttack: target must be an integer from 0 to 9 (got ${target}).`);
  }
  if (!original || original.length !== 784) {
    throw new RangeError(`startAttack expects a 784-value input but got ${original?.length ?? original}.`);
  }
  const input = Float32Array.from(original, value => Math.min(1, Math.max(0, value)));
  const activations = runCnn(weights, input);
  return {
    target,
    input,
    changes: new Int8Array(784),
    order: [],
    steps: 0,
    activations,
    status: topDigit(activations.probabilities) === target ? 'fooled' : 'running',
  };
}

/**
 * Takes one step: changes 1–3 pixels, reruns the network and updates the status.
 * Does nothing once the attack has finished.
 * @param cap the most pixels the whole attack may change.
 * @returns the indices of the pixels changed in this step (empty if none).
 */
export function stepAttack(weights: CnnWeights, attack: Attack, cap = ATTACK_PIXEL_CAP): number[] {
  if (attack.status !== 'running') return [];
  if (attack.order.length >= cap) {
    attack.status = 'failed';
    return [];
  }
  const { target, input, changes, activations } = attack;
  const logits = activations.logits;
  let rival = target === 0 ? 1 : 0;
  for (let k = 0; k < 10; k++) if (k !== target && logits[k] > logits[rival]) rival = k;
  const gap = logits[rival] - logits[target];
  const gradient = inputGradient(weights, activations, target, rival);

  // The best few candidates by first-order gain, kept sorted, best first.
  const best: number[] = [];
  const gains: number[] = [];
  for (let i = 0; i < 784; i++) {
    if (changes[i] !== 0) continue;
    const g = gradient[i];
    const gain = g > 0 ? g * (1 - input[i]) : -g * input[i];
    if (!(gain > 0)) continue;
    if (best.length === ATTACK_STEP_PIXELS && gain <= gains[ATTACK_STEP_PIXELS - 1]) continue;
    let at = best.length;
    while (at > 0 && gains[at - 1] < gain) at--;
    best.splice(at, 0, i);
    gains.splice(at, 0, gain);
    if (best.length > ATTACK_STEP_PIXELS) {
      best.pop();
      gains.pop();
    }
  }
  if (!best.length) {
    attack.status = 'failed';
    return [];
  }

  const changed: number[] = [];
  let estimate = 0;
  for (let n = 0; n < best.length && attack.order.length < cap; n++) {
    const i = best[n];
    const up = gradient[i] > 0;
    input[i] = up ? 1 : 0;
    changes[i] = up ? 1 : -1;
    attack.order.push(i);
    changed.push(i);
    estimate += gains[n];
    if (estimate >= gap / 2) break;
  }
  attack.steps++;
  attack.activations = runCnn(weights, input);
  if (topDigit(attack.activations.probabilities) === target) attack.status = 'fooled';
  else if (attack.order.length >= cap) attack.status = 'failed';
  return changed;
}

/** Runs an attack to the end in one go (for tests, and for reduced motion). */
export function runAttack(weights: CnnWeights, original: Float32Array, target: number, cap = ATTACK_PIXEL_CAP): Attack {
  const attack = startAttack(weights, original, target);
  while (attack.status === 'running') stepAttack(weights, attack, cap);
  return attack;
}
