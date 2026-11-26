/**
 * What the guide says. SHOTS names every screenshot build.mjs captures: the page
 * region to capture and the numbered markers placed on it, each with the note that
 * explains it. SLIDES is the guide itself, in order; shot slides number their notes
 * from the shot's markers, so a marker and its explanation can never drift apart.
 *
 * Regions: a selector, a list of selectors (their union), or an object
 * { sel, pad, padLeft, padTop, maxHeight }. 'viewport' captures the screen as is.
 * Marker anchors place the number: tl (default), tr, bl, br, l, r, t, c.
 */
import { activationGrid, attentionSvg, autoencoderDiagram, cnnPipelineSvg, colorLegend, learningRateSvg, lstmCellSvg, optimizerSvg, regularizationSvg, samplingSvg } from './diagrams.mjs';

export const SHOTS = {
  /* ---------- Site ---------- */
  header: {
    header: true,
    region: { sel: '.app-header', pad: 0 },
    markers: [
      { sel: 'button.brand', title: 'Neural Network Visualizer', text: 'Returns to the top of the page.' },
      { sel: '.main-nav', anchor: 't', title: 'One tab per model', text: 'Playground, Digits, Autoencoder, Recurrent and Transformer. Below 900 pixels wide the tabs show icons only.' },
      { sel: 'a.nav-button[aria-current="page"]', anchor: 'tr', title: 'The page you are on', text: 'Highlighted in white; clicking it scrolls back to the top. The Playground tab remembers your last experiment.' },
      { sel: '#theme-toggle', anchor: 'l', title: 'Light or dark', text: 'Follows your system setting until you choose, then remembers your choice. The colors keep their meaning in both themes.' },
      { sel: '#share-button', anchor: 'l', title: 'Share experiment', text: 'On the Playground only: copies a link that reopens the same settings, data and random seed.' },
    ],
  },
  'hero-playground': { header: true, region: 'viewport' },
  'pg-overview': {
    region: ['.workspace-intro', '#main-part'],
    markers: [
      { sel: '.page-links .text-button:nth-of-type(1)', title: 'How it works', text: 'Jumps to an illustrated guide below the workbench that explains what you are looking at.' },
      { sel: '.page-links .text-button:nth-of-type(2)', title: 'Classroom', text: 'For teachers: choose which controls students see, then share a link to that simpler view.' },
      { sel: '#top-controls', title: 'Training controls', text: 'Start, step and reset training, and choose how the network learns: learning rate, activation, optimizer and regularization.' },
      { sel: 'section:has(#data-heading)', title: 'Data', text: 'The problem, the dataset, and how much data, noise and batch size to train with.' },
      { sel: '.network-panel', title: 'Network', text: 'The network itself: input features on the left, hidden layers in the middle, the output on the right.' },
      { sel: 'section:has(#output-heading)', title: 'Output', text: 'How well it is learning: the losses, the loss over time, and its predictions over the whole plot.' },
    ],
  },
  'pg-training': {
    region: '#top-controls',
    markers: [
      { sel: '#play-pause-button', title: 'Run training, Pause', text: 'Trains one epoch after another (one epoch is a pass over all the training points) until you pause.' },
      { sel: '#next-step-button', title: 'Step', text: 'Trains exactly one epoch, so you can watch a single update.' },
      { sel: '#reset-button', title: 'Reset', text: 'Starts again from new random weights, keeping the data.' },
      { sel: '.epoch-counter', anchor: 'tr', title: 'Epoch counter', text: 'How many epochs have been trained, and whether training is running or paused.' },
      { sel: '.ui-learningRate', anchor: 'tr', title: 'Learning rate', text: 'How big a step the weights take on each update, from 0.00001 to 10.' },
      { sel: '.ui-activation', anchor: 'tr', title: 'Activation', text: 'The function every hidden neuron applies to its sum: Tanh, ReLU, Leaky ReLU, GELU, Sigmoid or Linear.' },
      { sel: '.ui-optimizer', anchor: 'tr', title: 'Optimizer', text: 'The rule that turns gradients into weight updates: SGD, Momentum or Adam.' },
      { sel: '.ui-regularization', anchor: 'tr', title: 'Regularization', text: 'An optional penalty on large weights, L1 or L2, against overfitting.' },
      { sel: '.ui-regularizationRate', anchor: 'tr', title: 'Regularization rate', text: 'How strong that penalty is. 0 switches it off.' },
    ],
  },
  'pg-data': {
    region: 'section:has(#data-heading)',
    markers: [
      { sel: '.ui-problem', anchor: 'ol', title: 'Problem type', text: 'Classification: every point is violet or amber and the network learns to tell them apart. Regression: every point has a value to predict.' },
      { sel: '.dataset-list .dataset:nth-child(1)', title: 'Datasets', text: 'Circle, XOR, Gaussian and Spiral. The outlined tile is in use. Regression offers Plane and Multi-Gauss instead.' },
      { sel: '.dataset-custom', title: 'Draw your own', text: 'Start from an empty plot and place your own violet and amber points.' },
      { sel: '.range-control.ui-percTrainData', anchor: 'ol', title: 'Training split', text: 'The share of points the network trains on (10 to 90%). The rest are test points it never learns from, used to check that it generalizes.' },
      { sel: '.range-control.ui-noise', anchor: 'ol', title: 'Noise', text: 'How far points are jittered from the clean pattern (0 to 50). More noise blurs the border between the classes.' },
      { sel: '.range-control.ui-batchSize', anchor: 'ol', title: 'Batch size', text: 'How many points are averaged into each weight update (1 to 30). Small batches give quick but noisy updates, large ones smoother but fewer.' },
      { sel: '#data-regen-button', title: 'Regenerate data', text: 'Draws a fresh random set of points with the same settings.' },
      { sel: '.dataset-meta', anchor: 'ol', title: 'Samples and seed', text: '500 points, made from a random seed so the same data comes back from a shared link.' },
    ],
  },
  'pg-features': {
    region: { sel: ['#canvas-x', '#canvas-sinY'], pad: 12, padLeft: 74 },
    markers: [
      { sel: '#canvas-x', anchor: 'r', title: 'x₁', text: 'The horizontal position: amber on the left, violet on the right.' },
      { sel: '#canvas-y', anchor: 'r', title: 'x₂', text: 'The vertical position.' },
      { sel: '#canvas-xSquared', anchor: 'r', title: 'x₁²', text: 'Large at the left and right edges, small in the middle.' },
      { sel: '#canvas-ySquared', anchor: 'r', title: 'x₂²', text: 'Large at the top and bottom.' },
      { sel: '#canvas-xTimesY', anchor: 'r', title: 'x₁x₂', text: 'Positive in two opposite corners and negative in the other two: exactly the XOR pattern.' },
      { sel: '#canvas-sinX', anchor: 'r', title: 'sin(x₁)', text: 'Waves from left to right.' },
      { sel: '#canvas-sinY', anchor: 'r', title: 'sin(x₂)', text: 'Waves from top to bottom.' },
    ],
  },
  'pg-network': {
    region: '.network-panel',
    markers: [
      { sel: '.network-summary', anchor: 'or', title: 'Hidden layers', text: 'How many layers sit between the inputs and the output (0 to 6). The buttons beside it add or remove one.' },
      { sel: '.plus-minus-neurons', title: 'Neurons per layer', text: '+ and − change how many neurons that layer has (1 to 8).' },
      { sel: '#canvas-x', anchor: 'tr', title: 'Input feature', text: 'Features that are on feed the first layer. Faded ones are off; click to toggle.' },
      { sel: '#canvas-1', anchor: 'tr', title: 'Neuron', text: 'Its output over the whole plot, in the output’s colors: the pattern this neuron has learned to detect. Hover to see it large in the Output panel.' },
      { sel: '#linkx-1', anchor: 'c', title: 'Connection, a weight', text: 'Color is its sign (violet positive, amber negative), thickness its size. Click one to type a new value.' },
      { sel: '#bias-1', anchor: 'bl', title: 'Bias', text: 'The small square at each neuron, added to its sum. Click to edit.' },
      { sel: '#canvas-7', anchor: 'tr', title: 'Output ŷ', text: 'The network’s answer over the whole plot, the same picture as the decision boundary.' },
      { sel: '.graph-callout', title: 'Margin notes', text: 'Short tips on what to try in the graph.' },
      { sel: '.parameter-badge', anchor: 'ol', title: 'Parameters', text: 'How many weights and biases the network has.' },
      { sel: 'label[for="show-backprop"]', anchor: 'ol', title: 'Show backpropagation', text: 'Shows how training is changing every weight (next pages).' },
      { sel: '.network-legend', anchor: 'ol', title: 'Legend', text: 'Violet lines are positive weights, amber negative.' },
    ],
  },
  'pg-backprop': {
    region: '.network-panel',
    markers: [
      { sel: 'label[for="show-backprop"]', anchor: 'ol', title: 'Show backpropagation', text: 'Gives every connection a twin line showing the last update.' },
      { sel: '#gradientx-1', anchor: 'c', title: 'Update line', text: 'Violet: the last update pushed this weight up. Amber: pushed it down. Thicker means a bigger push.' },
      { sel: '.backprop-legend', anchor: 'ol', title: 'Legend', text: 'The key for the update lines.' },
      { sel: '#canvas-7', anchor: 'tr', title: 'Where the error starts', text: 'While training, dashes flow backward from the output, the way the error travels. Step replays one update: the inputs light up toward the output, then the error runs back.' },
    ],
  },
  'pg-output': {
    region: 'section:has(#output-heading)',
    markers: [
      { sel: '.output-stats.train', anchor: 'tr', title: 'Training loss', text: 'How wrong the network is on its training points (mean squared error). Lower is better. The solid line in the chart.' },
      { sel: '.output-stats.ui-percTrainData', anchor: 'tr', title: 'Test loss', text: 'The same on the test points, which it never trains on. The dashed line in the chart.' },
      { sel: '#linechart', anchor: 'tr', title: 'Loss over time', text: 'Both losses after every epoch. If test loss turns up while training loss keeps falling, the network is overfitting.' },
      { sel: '#heatmap', title: 'Decision boundary', text: 'The network’s output at every position: violet where it says +1, amber where it says −1, pale where it is unsure. Dots are the training points in their true colors.' },
      { sel: '.color-key', title: 'Color scale', text: 'Amber is −1, white 0, violet +1. In regression the colors show the predicted value.' },
      { sel: '.ui-showTestData', anchor: 'ol', title: 'Show test data', text: 'Adds the test points as outlined dots.' },
      { sel: '.ui-discretize', anchor: 'ol', title: 'Discrete output', text: 'Shows only the predicted class, without shading for confidence.' },
      { sel: '#inspection-label', anchor: 'ol', title: 'What the plot shows', text: '“Output” is the network’s answer. Hover a neuron and the plot shows that neuron instead.' },
    ],
  },
  'pg-datasets-classification': { region: { sel: '.dataset-list', pad: 6 } },
  'pg-datasets-regression': { region: { sel: '.dataset-list', pad: 6 } },
  'pg-noise-0': { region: { sel: '#heatmap', pad: 4 } },
  'pg-noise-40': { region: { sel: '#heatmap', pad: 4 } },
  'pg-draw': {
    region: '#main-part',
    markers: [
      { sel: '.dataset-custom', title: 'Draw your own', text: 'Choose this tile to start an empty dataset.' },
      { sel: '#paint-controls', title: 'Paint', text: 'Pick violet, amber or erase, then click the plot to place or remove points.' },
      { sel: '#heatmap', anchor: 'tr', title: 'The plot', text: 'Your points appear here, and the network keeps training as you add them.' },
      { sel: '#clear-points', anchor: 'ol', title: 'Clear points', text: 'Removes them all. Your points travel in the share link, up to 400.' },
    ],
  },
  'pg-guide': { region: { sel: '#article-text', maxHeight: 900 } },
  'pg-classroom': { region: '#classroom-panel' },

  /* ---------- Digits ---------- */
  'hero-digits': { header: true, region: 'viewport' },
  'cnn-overview': {
    region: '#cnn-workbench',
    markers: [
      { sel: '#draw-pad', title: 'Scratchpad', text: 'Draw any digit 0 to 9 with a mouse, finger or pen. It is read live as you draw.' },
      { sel: '#clear-pad', title: 'Clear', text: 'Erases the pad.' },
      { sel: '#example-digit', title: 'Try an example', text: 'Loads a real handwritten digit from the MNIST test set.' },
      { sel: '#input-view', title: 'What the network sees', text: 'Your drawing scaled into 20 × 20 pixels and centered in 28 × 28, exactly like the training data.' },
      { sel: '#saliency-view', title: 'What mattered most', text: 'Which pixels drove the answer: violet for, amber against.' },
      { sel: '#conv1-maps', title: 'Layers', text: 'The output of every layer: two convolutions, two poolings and a dense layer.' },
      { sel: '#map-inspector', title: 'Map inspector', text: 'The feature map you point at, large, with the filter weights that made it.' },
      { sel: '#verdict', title: 'Prediction', text: 'The digit with the highest probability, and how sure the network is.' },
      { sel: '#digit-bars', title: 'Probabilities', text: 'The network’s probability for each digit 0 to 9. They add up to 100%.' },
      { sel: '#fool-target', title: 'Fool the network', text: 'Pick another digit and watch a few changed pixels trick the network.' },
    ],
  },
  'cnn-layers': {
    region: 'section:has(#layers-heading)',
    markers: [
      { sel: '#conv1-maps', title: 'Convolution 1', text: '8 feature maps. Each shows where its 3 × 3 filter found its pattern, such as an edge or a curve; stronger violet is a stronger response.' },
      { sel: '#pool1-maps', title: 'Max pooling', text: 'The same 8 maps at half size: each keeps the strongest value of every 2 × 2 patch.' },
      { sel: '#conv2-maps', title: 'Convolution 2', text: '16 maps, each combining all 8 previous maps to find larger shapes.' },
      { sel: '#pool2-maps', title: 'Max pooling', text: '16 maps of 7 × 7: 784 numbers in all.' },
      { sel: '#dense-view', title: 'Dense', text: '64 neurons, each looking at all 784 numbers. Darker violet means more active.' },
      { sel: '#inspector-map', title: 'The selected map', text: 'Hover or focus any map to show it here, large.' },
      { sel: '#inspector-kernels', title: 'Its filter weights', text: 'The 3 × 3 weights behind that map: violet weights add the pixel underneath, amber ones subtract it. A second-layer map has one 3 × 3 filter per input map, 8 in all.' },
      { sel: '#parameter-total', anchor: 'ol', title: 'Parameters', text: 'The network has 52,138 trainable numbers.' },
    ],
  },
  'cnn-prediction': {
    region: 'section:has(#prediction-heading)',
    markers: [
      { sel: '#predicted-digit', title: 'The answer', text: 'The digit with the highest probability.' },
      { sel: '#predicted-confidence', title: 'How sure', text: 'That digit’s probability.' },
      { sel: '#digit-bars', title: 'All ten probabilities', text: 'Softmax turns the network’s ten scores into probabilities that add up to 100%. The darkest bar is the answer.' },
      { sel: '#fool-target', title: 'Target', text: 'The digit to turn the drawing into.' },
      { sel: '#fool-start', title: 'Fool it', text: 'Starts changing pixels.' },
    ],
  },
  'cnn-saliency': {
    region: { sel: '.pixel-views', pad: 8 },
    markers: [
      { sel: '#input-view', title: 'What the network sees', text: 'The 28 × 28 input.' },
      { sel: '#saliency-view', title: 'What mattered most', text: 'Gradient × input: how much each pixel’s ink changed the score of the answer. Violet ink pushed toward the answer, amber ink against it; uncolored pixels hardly mattered.' },
    ],
  },
  'cnn-fooled': {
    region: '#cnn-workbench',
    markers: [
      { sel: '#input-view', title: 'Changed pixels', text: 'The pixels the attack changed are outlined. Each step changes the one pixel that helps the target most.' },
      { sel: '#predicted-digit', title: 'Fooled', text: 'The network now reads the target digit.' },
      { sel: '#fool-status', anchor: 'ol', title: 'How many', text: 'How many pixels it took, usually about 18 of 784.' },
      { sel: '#fool-undo', anchor: 'ol', title: 'Undo', text: 'Brings back your drawing.' },
    ],
  },

  /* ---------- Autoencoder ---------- */
  'hero-vae': { header: true, region: 'viewport' },
  'vae-overview': {
    region: '#vae-workbench',
    markers: [
      { sel: '#map-canvas', title: 'The map', text: 'Every point is a pair of numbers: z₁ across, z₂ up. This is the latent space.' },
      { sel: '.segmented', anchor: 'ol', title: 'Map view', text: 'Test digits, or a grid of decoded digits.' },
      { sel: '#probe', anchor: 'r', title: 'The point', text: 'Drag it, tap anywhere on the map, or use the arrow keys (Shift for bigger steps).' },
      { sel: '.map-key', anchor: 'ol', title: 'Dots', text: 'Each grey dot is one of 5,000 test digits the network never trained on, placed where the encoder puts it. Numerals mark where each digit is most common.' },
      { sel: '.class-row', anchor: 'ol', title: 'Go to', text: 'Jumps to a digit’s cluster. Hovering a button or a dot turns that whole class violet.' },
      { sel: '#decoded-view', title: 'At the point', text: 'The 28 × 28 digit the decoder draws from the point’s two numbers, redrawn as you move.' },
      { sel: '#z1', anchor: 'ol', title: 'Coordinates', text: 'The point’s two numbers.' },
      { sel: '#looks-like', anchor: 'ol', title: 'Looks like', text: 'The Digits page’s network reading the decoded image, and how sure it is.' },
      { sel: '#morph-from', title: 'Morph', text: 'Choose From and To (a digit, the point or your drawing), then press Morph to slide between them.' },
      { sel: '#morph-strip', title: 'Steps', text: 'The digits along the way. Each is decoded from its own position, not two pictures faded together.' },
      { sel: '#draw-pad', title: 'Scratchpad', text: 'Draw a digit, or try an example, to see where it lands on the map.' },
      { sel: '.pixel-views', anchor: 'ol', title: 'Encoder and decoder', text: 'Your drawing as 28 × 28 pixels, and the decoder’s version redrawn from just two numbers.' },
    ],
  },
  'vae-grid': { region: 'section:has(#map-heading)' },
  'vae-example': {
    region: '#vae-workbench',
    markers: [
      { sel: '#draw-pad', title: 'Your drawing', text: 'Draw a digit, or press Try an example. It is encoded as you draw.' },
      { sel: '#drawing-readout', anchor: 'ol', title: 'Where it lands', text: 'Its five nearest test digits on the map vote on where it landed.' },
      { sel: '.pixel-views', anchor: 'ol', title: 'In and out', text: 'What the encoder read, and what the decoder draws back from just two numbers.' },
      { sel: '#map-canvas', title: 'On the map', text: 'The point jumps to the drawing’s two numbers, and a dashed ring marks the spot.' },
      { sel: '#morph-from', title: 'Your drawing', text: 'Now a choice for From and To, so you can morph from your own digit.' },
    ],
  },
  'vae-input': { region: { sel: '#input-view', pad: 0 } },
  'vae-output': { region: { sel: '#reconstruction-view', pad: 0 } },

  /* ---------- Recurrent ---------- */
  'hero-rnn': { header: true, region: 'viewport' },
  'rnn-overview': {
    region: { sel: ['.prompt-dock', '#rnn-workbench'], maxHeight: 1080 },
    markers: [
      { sel: '#prompt-input', anchor: 'tr', title: 'Text', text: 'Type anything, up to 400 characters. Every view updates as you type.' },
      { sel: '#example-select', anchor: 'ob', title: 'Examples', text: 'Sample sentences, including dialogue with quotation marks.' },
      { sel: '#generate-button', anchor: 'ob', title: 'Generate', text: 'Writes 120 more characters, about 30 a second, each picked from the five likeliest. Press Stop to end early.' },
      { sel: '.dock-range', anchor: 'ob', title: 'Temperature', text: 'How adventurous Generate is (0.1 to 2; 0.6 by default). Lower is safer and more repetitive, higher more surprising.' },
      { sel: '.memory-panel', title: 'Memory', text: 'What the network holds in memory at every character (next pages).' },
      { sel: '.next-panel', anchor: 'tr', title: 'Next character', text: 'What it expects after the selected character, and how often it is right.' },
    ],
  },
  'rnn-memory-top': {
    region: { sel: ['#memory-heading', '#cell-map'], pad: 16, within: '.memory-panel' },
    markers: [
      { sel: '.surprise-key .callout-note', anchor: 'ol', title: 'Steps', text: 'Each box is one step: a character read. Hover, tap or use the arrow keys on a box to select it.' },
      { sel: '#span-summary', textBox: true, anchor: 'or', title: 'Memory span', text: 'For the selected character: how much each earlier character shaped the guess (gradient × input), and how many of the latest characters hold half and 90% of it.' },
      { sel: '#span-bars', anchor: 'ol', title: 'Span bars', text: 'Violet bars pushed toward the guess, amber against. Watch them fade with distance.' },
      { sel: '#char-cells', anchor: 'ol', title: 'Characters', text: 'One box per character. The amber tint shows how surprising it was; the outlined box is the selected step.' },
      { sel: '.surprise-legend', anchor: 'ol', title: 'Surprise scale', text: 'Stronger amber is more surprising: no tint below 1 bit, strongest from 8.' },
      { sel: '#cell-map', anchor: 'ol', title: 'Cell state c', text: 'The memory: one row per memory cell (256), one column per character. Violet above zero, amber below. The outlined row is the traced cell.' },
    ],
  },
  'rnn-memory-bottom': {
    region: { sel: ['#unit-title', '#map-readout'], pad: 16, padBottom: 6, within: '.memory-panel' },
    markers: [
      { sel: '#unit-title', anchor: 'ol', title: 'The traced cell', text: 'One cell’s memory over the text: violet area above zero, amber below. Hover a map row, or use the arrow keys on a map, to trace another cell.' },
      { sel: '#cell-picks', anchor: 'or', title: 'Try', text: 'Cells that learned something you can see: the quotation cell changes sign inside quotes; the word-length cell counts letters.' },
      { sel: '.gate-strips', anchor: 'ol', title: 'Its gates', text: 'The traced cell’s input, forget and output gates at every step, from 0 (pale) to 1 (violet).' },
      { sel: '.detail-label .segmented', anchor: 'ol', title: 'Second map', text: 'Switch between the hidden state and the three gates of every cell.' },
      { sel: '#detail-map', anchor: 'ol', title: 'The second map', text: 'Gates run from 0 (pale) to 1 (violet); the hidden state from −1 (amber) to +1 (violet).' },
      { sel: '#map-readout', anchor: 'ol', title: 'Readout', text: 'The traced cell’s exact values at the selected step.' },
    ],
  },
  'rnn-next': {
    width: 820,
    region: '.next-panel',
    markers: [
      { sel: '#next-context', anchor: 'ol', title: 'Context', text: 'The text up to the selected character.' },
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-text', anchor: 'ol', title: 'Candidates', text: 'The ten likeliest next characters, likeliest first.' },
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-word', textBox: true, anchor: 'or', title: 'Word preview', text: 'The word each letter would lead to if the network kept taking its likeliest letter. The grey part is already written. Shown for candidates of 1% or more.' },
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-value', anchor: 'or', title: 'Probability', text: 'After temperature.' },
      { sel: '#candidate-list .candidate.is-filtered', anchor: 'ol', title: 'Dimmed rows', text: 'Generate only picks from the five likeliest.' },
      { sel: '#candidate-list .candidate.is-actual', anchor: 'ol', title: 'What came next', text: 'The character the text really has next is outlined. The sentence below gives its rank.' },
      { sel: '#scorecard', anchor: 'ol', title: 'How often it’s right', text: 'On this text: the share of characters it guessed first, and within three guesses. Inside a word it is usually right; at the start of a word, where it must pick the next word, much less often.' },
    ],
  },
  'rnn-gates-top': {
    region: { sel: ['#gates-heading', '#quote-chart'], pad: 14 },
    markers: [
      { sel: '#gates-lstm-note', textBox: true, anchor: 'or', title: 'The LSTM', text: 'Its guess for the selected character, and how far back its memory span reaches.' },
      { sel: '#gates-lstm-chars', anchor: 'ol', title: 'Its surprise', text: 'Each character tinted by how surprising it was to the LSTM.' },
      { sel: '#gates-rnn-note', textBox: true, anchor: 'or', title: 'The gateless network', text: 'The same for a recurrent network with about as many weights (512 memory numbers) but no gates.' },
      { sel: '#gates-rnn-chars', anchor: 'ol', title: 'Its surprise', text: 'Compare the tints character by character.' },
      { sel: '#quote-note', textBox: true, anchor: 'or', title: 'Inside a quotation', text: 'The LSTM’s quotation cell (violet) and the gateless network’s closest unit (dashed), drawn so up means inside the shaded quotes.' },
      { sel: '#quote-chart', anchor: 'tr', title: 'The difference', text: 'The LSTM’s line switches cleanly at every quote; the gateless one jitters.' },
    ],
  },
  'rnn-gates-bottom': {
    region: { sel: ['#fade-chart', '#gates-scores', '#continue-results'], pad: 14, padTop: 22, padBottom: 6 },
    markers: [
      { sel: '.fade-figure .gates-title', textBox: true, anchor: 'or', title: 'How fast memory fades', text: 'Each earlier character’s average share of the guess, by how far back it is, on a log scale. LSTM violet and solid, gateless dashed; pale curves are held-out text.' },
      { sel: '#fade-summary', anchor: 'ol', title: 'In numbers', text: 'Up close the two lean on recent characters alike. Beyond about 50 characters the gateless share falls about 6% per character, while the LSTM keeps a long tail: 10 times more at 100 back.' },
      { sel: '#gates-scores caption', textBox: true, anchor: 'or', title: 'How often each is right', text: 'First guess and within three, on this text and on the held-out ends of the books.' },
      { sel: '#quote-evidence .gates-title', textBox: true, anchor: 'or', title: 'Keeping track of a quotation', text: 'How closely each network’s best unit follows being inside a quote: r = −0.74 for the LSTM, −0.29 without gates.' },
      { sel: '#continue-button', anchor: 'ol', title: 'Write with both', text: 'Each network continues your text, side by side.' },
    ],
  },

  /* ---------- Transformer ---------- */
  'tf-loader': { region: '#model-loader' },
  'hero-transformer': { header: true, region: 'viewport' },
  'tf-dock': {
    region: '.prompt-dock',
    markers: [
      { sel: '#prompt-input', anchor: 'tr', title: 'Prompt', text: 'Any text; the first 48 tokens are shown. Line breaks count as tokens too.' },
      { sel: '#surprise', anchor: 'ol', title: 'Surprising words', text: 'Each token tinted by how unexpected it was to GPT-2.' },
      { sel: '#example-select', anchor: 'ob', title: 'Examples', text: 'Sample prompts, including the Mary and John sentence used on these pages.' },
      { sel: '#generate-button', anchor: 'ob', title: 'Generate', text: 'Draws one token with the settings to the right and appends it.' },
      { sel: '.range-control.dock-range:has(#temperature)', anchor: 'ob', title: 'Temperature', text: 'Below 1 sharpens the probabilities toward the top token; above 1 flattens them.' },
      { sel: '.sampling-control', anchor: 'ob', title: 'Sampling', text: 'Top-k keeps the k likeliest tokens; top-p keeps the fewest whose probabilities add up to p.' },
      { sel: '#engine-name', anchor: 'ol', title: 'Where it runs', text: 'On your GPU through WebGPU, or on the CPU, with the time the last run took.' },
      { sel: '#engine-choice', anchor: 'ol', title: 'Run on', text: 'Switch between GPU and CPU to compare speeds. The results agree.' },
    ],
  },
  'tf-overview': {
    region: { sel: ['.prompt-dock', '#transformer-flow'], maxHeight: 1040 },
    markers: [
      { sel: 'section:has(#embedding-heading)', title: 'Embedding', text: 'The prompt as tokens, and the vectors GPT-2 starts from.' },
      { sel: 'section:has(#blocks-heading)', title: 'Transformer blocks', text: '12 blocks, each with 12 attention heads and an MLP. Choose a block and a head to look inside.' },
      { sel: 'section:has(#probability-heading)', title: 'Next token', text: 'The probabilities for the next token, how the guess forms, and which heads matter.' },
    ],
  },
  'tf-embedding-tokens': {
    region: { sel: ['section:has(#embedding-heading) .panel-heading', 'section:has(#embedding-heading) .callout-note'], pad: 16 },
    markers: [
      { sel: '#token-list .token:nth-child(2) .token-text', anchor: 'ol', title: 'Tokens', text: 'GPT-2 reads text as tokens: whole words, pieces of words or punctuation. A dot before a token means it starts with a space.' },
      { sel: '#token-list .token:nth-child(2) .token-id', anchor: 'ol', title: 'Token ID', text: 'Its number in GPT-2’s vocabulary of 50,257 tokens.' },
      { sel: '#token-list .token:nth-child(2) .token-attention', anchor: 'or', title: 'Attention bar', text: 'How much the focused token (the last one, or the one you hover) attends to this token in the chosen head.' },
    ],
  },
  'tf-embedding-maps': {
    region: { sel: '.embedding-stack', pad: 16 },
    markers: [
      { sel: '#token-embedding', anchor: 'c', title: 'Token embedding', text: '768 learned numbers for each token, one row per token: violet positive, amber negative.' },
      { sel: '#position-embedding', anchor: 'c', title: 'Position embedding', text: '768 numbers that tell the model where the token sits.' },
      { sel: '#input-embedding', anchor: 'c', title: 'Input to block 1', text: 'The sum of the two: the vector each token carries into the first block.' },
    ],
  },
  'tf-blocks': {
    region: { sel: ['#block-strip', '#head-grid', '#block-toggle', '#head-toggle'], pad: 14 },
    markers: [
      { sel: '#block-strip', title: 'The 12 blocks', text: 'Each thumbnail is the block’s average attention over its 12 heads: rows are tokens that look, columns the earlier tokens they look at, violet means more attention.' },
      { sel: '#block-strip button:nth-child(12)', anchor: 'tr', title: 'Open block', text: 'Outlined. Choose any block to open it.' },
      { sel: '#head-grid', title: 'Its 12 heads', text: 'Each thumbnail is that head’s own attention pattern.' },
      { sel: '#head-grid button:nth-child(11)', anchor: 'tr', title: 'Open head', text: 'Outlined; shown in detail to the right.' },
      { sel: '#block-toggle', title: 'Switch off block', text: 'Switches off all 12 heads of this block at once.' },
      { sel: '#head-title', anchor: 'ol', title: 'Block and head', text: 'Which head is shown below.' },
      { sel: '#head-toggle', anchor: 'tr', title: 'Switch off this head', text: 'See what the head contributes (later pages).' },
    ],
  },
  'tf-head': {
    region: { sel: ['#head-title', '#attention-readout'], pad: 14 },
    markers: [
      { sel: '#query-matrix', anchor: 'c', title: 'Query, Q = X·W_Q', text: 'What each token is looking for: 64 numbers per token.' },
      { sel: '#key-matrix', anchor: 'c', title: 'Key, K = X·W_K', text: 'What each token offers to be found by.' },
      { sel: '#value-matrix', anchor: 'c', title: 'Value, V = X·W_V', text: 'What each token passes on when it is attended to.' },
      { sel: '.weights-figure', anchor: 'c', title: 'This head’s weights', text: 'W_Q, W_K and W_V: 768 inputs across, 64 outputs down. The learned matrices that make Q, K and V.' },
      { sel: '#score-matrix', anchor: 'c', title: 'Scores', text: 'Q·Kᵀ / √64: how well each query matches every earlier key. Hatched cells are future tokens, masked so no token can look ahead.' },
      { sel: '#attention-matrix', anchor: 'c', title: 'Attention', text: 'Softmax of each row: scores turned into shares that add up to 1. Violet means more attention.' },
      { sel: '#head-output', anchor: 'c', title: 'Output', text: 'Attention × V: each token’s mix of the values it attended to.' },
      { sel: '#attention-readout', anchor: 'ol', title: 'In words', text: 'Here: “to” gives 52.8% of its attention to “Mary”.' },
    ],
  },
  'tf-mlp': {
    region: { sel: ['.mlp-figure'], pad: 14 },
    markers: [
      { sel: '#mlp-matrix', title: 'MLP', text: 'After attention, each token’s vector goes through a two-layer network on its own: 768 → 3,072 neurons with GELU → 768. One row per token, one column per neuron.' },
    ],
  },
  'tf-next': {
    width: 820,
    region: { sel: ['#probability-heading', 'figure:has(#final-hidden)'], pad: 18 },
    markers: [
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-text', anchor: 'ol', title: 'Candidates', text: 'The likeliest next tokens with their probabilities, after temperature.' },
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-value', anchor: 'or', title: 'Probability', text: 'The chance Generate picks it, after the sampling filter.' },
      { sel: '#candidate-list .candidate.is-filtered', anchor: 'ol', title: 'Filtered out', text: 'Grey tokens are outside the top-k or top-p filter, so Generate never picks them.' },
      { sel: '#sampling-summary', anchor: 'ol', title: 'Sampling summary', text: 'How many tokens are kept, and their share of the probability.' },
      { sel: 'figure:has(#final-hidden)', anchor: 'ol', title: 'Final hidden state', text: 'After 12 blocks, the last token’s 768 numbers. They are scored against all 50,257 token embeddings to give the next-token scores.' },
    ],
  },
  'tf-lens': {
    width: 820,
    region: '#lens',
    markers: [
      { sel: '#lens-context', anchor: 'ol', title: 'Which prediction', text: 'What could follow the focused token (the last one, or the one you hover).' },
      { sel: '#lens-list li:nth-child(1) .lens-row', anchor: 'ol', title: 'After the embedding', text: 'What GPT-2 would predict straight from the input, before any block: usually nonsense.' },
      { sel: '#lens-list li:nth-child(10) .lens-row', anchor: 'ol', title: 'Middle blocks', text: 'The guess forming. Here “them” leads after blocks 9 and 10.' },
      { sel: '#lens-list li:last-child .lens-row', anchor: 'ol', title: 'Block 12', text: 'The model’s real prediction: “Mary”, 45.3%. Click any row to open that block.' },
    ],
  },
  'tf-surprise': {
    region: { sel: ['#prompt-input', '#surprise'], pad: 12, padTop: 38 },
    markers: [
      { sel: '#surprise-text span:nth-child(2)', title: 'A surprising token', text: '“Mary” right after “When” was hard to predict: strong amber.' },
      { sel: '#surprise-text span:last-child', anchor: 'tr', title: 'An expected token', text: '“to” after “gave a drink” was easy: no tint.' },
      { sel: '.surprise-legend', anchor: 'l', title: 'Scale', text: 'No tint below 2 bits; the strongest from 14. Hover a token for its exact bits and rank.' },
    ],
  },
  'tf-off': {
    region: { sel: ['#block-strip', '#attention-readout'], pad: 14 },
    markers: [
      { sel: '#block-strip button:nth-child(12)', anchor: 'tr', title: 'Block badge', text: 'How many heads are switched off in that block.' },
      { sel: '#head-grid button:nth-child(11)', anchor: 'tr', title: 'Switched-off head', text: 'A slash and a struck-through number.' },
      { sel: '#head-state', anchor: 'ot', title: 'Switched off', text: 'The badge beside the head’s name.' },
      { sel: '#head-toggle', anchor: 'or', title: 'Switch it back on', text: 'Restores this head.' },
      { sel: '#head-output', anchor: 'c', title: 'Output: all zeros', text: 'The head still computes its attention, but passes nothing on: its output is replaced by zeros and everything after it is recomputed.' },
      { sel: '#attention-readout', anchor: 'ol', title: 'In words', text: '“to” still gives 52.8% of its attention to “Mary”, but none of it is passed on.' },
    ],
  },
  'tf-off-next': {
    width: 820,
    region: { sel: ['#probability-heading', '#candidate-list'], pad: 18, padBottom: 4 },
    markers: [
      { sel: '#ablation-summary', anchor: 'ol', title: 'What changed', text: 'The biggest change in one sentence.' },
      { sel: '#comparison-key', anchor: 'ol', title: 'Key', text: 'While heads are off, values are before temperature and sampling. Violet is probability gained, hatched amber lost, since every head was on.' },
      { sel: '#candidate-list .candidate:nth-child(1) .candidate-value', anchor: 'or', title: 'Before → after', text: '“Mary”: 45.3% with every head on, 73.3% now.' },
      { sel: '#candidate-list .candidate:nth-child(2) .candidate-text', anchor: 'ol', title: 'A loser', text: '“them” fell from 20.8% to 8.4%.' },
    ],
  },
  'tf-importance': {
    width: 820,
    region: '#importance',
    markers: [
      { sel: '#importance-start', anchor: 'ol', title: 'Test every head', text: 'Switches off each of the 144 heads on its own and measures the top guess. About a second on a GPU; on the CPU it tests the open block’s 12.' },
      { sel: '#importance-progress', anchor: 'or', title: 'Time taken', text: '' },
      { sel: '#importance-grid button.importance-cell:nth-of-type(1)', anchor: 'l', title: 'Block 1, head 1: amber', text: 'Switching it off lowers the top guess: it was helping. Without it “Mary” falls from 45.3% to 10.9%.' },
      { sel: '#importance-grid button.importance-cell:nth-of-type(143)', anchor: 'tr', title: 'Block 12, head 11: violet', text: 'Switching it off raises the top guess: it was holding it back. Without it “Mary” rises to 73.3%.' },
      { sel: '.importance-legend', anchor: 'ol', title: 'Legend', text: 'Rows are blocks 1 to 12, columns heads 1 to 12. Pale cells hardly matter.' },
      { sel: '#importance-readout', anchor: 'ol', title: 'Summary', text: 'The strongest helper and holder-back in words. Click any cell to open that head.' },
    ],
  },

  /* ---------- Site-wide ---------- */
  'theme-light': { header: true, region: 'viewport' },
  'theme-dark': { header: true, region: 'viewport' },
  'phone-playground': { header: true, region: 'viewport', phone: true },
  'phone-vae': { header: true, region: 'viewport', phone: true },
  'phone-rnn': { header: true, region: 'viewport', phone: true },
};

const img = name => `{{img:${name}}}`;

export const SLIDES = [
  { type: 'cover' },
  {
    type: 'text', section: 'Introduction', title: 'Five neural networks you can look inside',
    lead: 'Each page runs a real neural network in your browser and shows what happens inside it. Nothing you type or draw is sent anywhere.',
    html: `<div class="cards">
      <div class="card"><img src="{{img:hero-playground}}" alt=""><strong>Playground</strong><em>A small network you build and train</em><span>Train it on 2D points and watch how layers, neurons, activations and training settings change what it learns.</span></div>
      <div class="card"><img src="{{img:hero-digits}}" alt=""><strong>Digits</strong><em>A convolutional network (CNN)</em><span>Reads a digit you draw, showing every filter and layer, the pixels that mattered, and how a few pixels can fool it.</span></div>
      <div class="card"><img src="{{img:hero-vae}}" alt=""><strong>Autoencoder</strong><em>A variational autoencoder (VAE)</em><span>Squeezes every digit into two numbers, so all digits sit on one map. Drag a point and watch digits morph.</span></div>
      <div class="card"><img src="{{img:hero-rnn}}" alt=""><strong>Recurrent</strong><em>A long short-term memory network (LSTM)</em><span>Reads text one character at a time, carrying a memory. Shows its memory cells, its gates and how far back it remembers.</span></div>
      <div class="card"><img src="{{img:hero-transformer}}" alt=""><strong>Transformer</strong><em>GPT-2 small, 124 million parameters</em><span>The real language model: tokens, embeddings, all 144 attention heads, and how it chooses the next word.</span></div>
    </div>`,
  },
  { type: 'shot', section: 'Introduction', title: 'Getting around', lead: 'Every page shares the same header.', shot: 'header', layout: 'stack' },
  {
    type: 'text', section: 'Introduction', title: 'What the colors mean',
    lead: 'The whole site uses one color language. Violet and amber always carry meaning; everything else stays neutral.',
    html: colorLegend(),
  },

  /* ---------- Playground ---------- */
  {
    type: 'divider', section: 'Playground', title: 'Playground', shot: 'hero-playground',
    lead: 'Build a small neural network, train it on 2D points, and watch every neuron learn. It follows Google’s TensorFlow Playground, with extras: Momentum and Adam, Leaky ReLU and GELU, a backpropagation view and hand-drawn data.',
    points: ['Each point has two coordinates, x₁ and x₂, and a color: violet (+1) or amber (−1).', 'The network learns to predict the color of any position, so its answer can be drawn over the whole plot.'],
  },
  { type: 'shot', section: 'Playground', title: 'The playground at a glance', shot: 'pg-overview' },
  { type: 'shot', section: 'Playground', title: 'Training controls', shot: 'pg-training', layout: 'stack' },
  {
    type: 'text', section: 'Playground', title: 'Learning rate',
    lead: 'Training repeats one move: work out which way each weight should change to lower the loss (the gradient), then step every weight a little that way. The learning rate is the size of that step.',
    html: `${learningRateSvg()}
      <div class="columns three">
        <p><strong>Too small</strong> (0.00001 to 0.001): every step is tiny, so the loss falls very slowly.</p>
        <p><strong>About right</strong> (0.01 to 0.1; 0.03 by default): the loss falls quickly and settles.</p>
        <p><strong>Too large</strong> (1 to 10): steps overshoot the bottom and bounce ever higher. The loss explodes and the page says training diverged; lower the rate and reset.</p>
      </div>`,
  },
  {
    type: 'text', section: 'Playground', title: 'Activation functions',
    lead: 'Each hidden neuron adds up its weighted inputs and its bias, then passes the sum through an activation function. The bend in that function is what lets a network draw curved boundaries.',
    html: `${activationGrid()}<p class="footnote">The output neuron uses tanh for classification (−1 to +1) and a linear output for regression.</p>`,
  },
  {
    type: 'text', section: 'Playground', title: 'Optimizers',
    lead: 'The optimizer turns gradients into weight updates. Below, each one walks down the same long, narrow valley of loss toward its lowest point.',
    html: `${optimizerSvg()}
      <div class="columns three">
        <p><strong>SGD</strong>, stochastic gradient descent: every step is the learning rate times the gradient. Simple, and the original playground’s rule, but it zigzags across narrow valleys.</p>
        <p><strong>Momentum</strong> keeps a running velocity, 90% of the last step plus the new gradient, so steps build up in consistent directions and zigzags cancel out.</p>
        <p><strong>Adam</strong> keeps running averages of each weight’s gradient and squared gradient, giving every weight its own step size. Usually fastest, and least sensitive to the learning rate.</p>
      </div>
      <p class="footnote">Paths simulated with each optimizer’s update rule on the valley ½(x² + 25y²).</p>`,
  },
  {
    type: 'text', section: 'Playground', title: 'Regularization and overfitting',
    lead: 'With enough weights, a network can memorize its training points, noise and all, instead of learning the pattern. That is overfitting: training loss keeps falling while test loss rises. Regularization adds a penalty for large weights.',
    html: `${regularizationSvg()}
      <div class="columns three">
        <p><strong>L1</strong> adds rate × |w| for every weight. It pushes small weights all the way to zero, switching off whole connections and leaving a sparser network.</p>
        <p><strong>L2</strong> adds rate × w². It shrinks every weight smoothly toward zero, keeping all of them small.</p>
        <p><strong>Regularization rate</strong> (0 to 10) sets the strength. Too strong and the network can’t fit the data at all, which is underfitting.</p>
      </div>`,
  },
  { type: 'shot', section: 'Playground', title: 'Data', shot: 'pg-data' },
  {
    type: 'images', section: 'Playground', title: 'Datasets and noise',
    lead: 'The four classification patterns, the two regression ones, and the same circle with and without noise.',
    images: [
      { shot: 'pg-datasets-classification', caption: 'Circle, XOR, Gaussian, Spiral and your own.' },
      { shot: 'pg-datasets-regression', caption: 'Regression: Plane and Multi-Gauss.' },
      { shot: 'pg-noise-0', caption: 'Noise 0: a clean border.' },
      { shot: 'pg-noise-40', caption: 'Noise 40: the classes overlap.' },
    ],
    notes: [
      ['Circle', 'One class in the middle, the other in a ring around it: needs a curved boundary.'],
      ['XOR', 'Violet where x₁ and x₂ have the same sign, amber where they differ. No straight line can split it.'],
      ['Gaussian', 'Two separate blobs. Even one straight line works.'],
      ['Spiral', 'Two interleaved spirals: the hardest pattern, needing more layers or the sine features.'],
      ['Plane and Multi-Gauss', 'Regression: smooth surfaces of values instead of two classes.'],
      ['Noise', 'With noise no boundary can be perfect. A good network learns the smooth shape underneath rather than chasing stray points.'],
    ],
  },
  {
    type: 'shot', section: 'Playground', title: 'Input features', shot: 'pg-features', imageWidth: 300,
    lead: 'Each input is a feature computed from a point’s position. Its square shows the feature’s value over the plot: violet where positive, amber where negative. Click one to switch it on or off.',
    after: 'Good features do the network’s work for it: with x₁² and x₂², even a network with no hidden layer can separate the circle.',
  },
  { type: 'shot', section: 'Playground', title: 'The network', shot: 'pg-network' },
  {
    type: 'shot', section: 'Playground', title: 'Watching backpropagation', shot: 'pg-backprop',
    lead: 'Backpropagation works out, for every weight, which way and how much to change it to lower the loss. Show backpropagation draws the last update on every connection.',
  },
  { type: 'shot', section: 'Playground', title: 'Output', shot: 'pg-output' },
  { type: 'shot', section: 'Playground', title: 'Draw your own data', shot: 'pg-draw' },
  {
    type: 'images', section: 'Playground', title: 'Help for learners and teachers', layout: 'stack',
    images: [
      { shot: 'pg-guide', caption: 'How it works: an illustrated guide below the workbench.' },
      { shot: 'pg-classroom', caption: 'Classroom: choose which controls students see.' },
    ],
    notes: [
      ['How it works', 'Explains the data, the network and the output in plain words, with ideas for experiments.'],
      ['Classroom', 'Tick the controls to show, then apply the view or copy a link that opens it. Useful for focused exercises.'],
    ],
  },

  /* ---------- Digits ---------- */
  {
    type: 'divider', section: 'Digits', title: 'Digits', shot: 'hero-digits',
    lead: 'Draw a digit and watch a convolutional neural network read it, layer by layer. It was trained for this site on MNIST, 70,000 handwritten digits, and reads 99.4% of the 10,000 test digits correctly.',
    points: ['Every intermediate result is shown: feature maps, pooling, the dense layer and the probabilities.', 'It also shows which pixels mattered, and how few changed pixels it takes to fool it.'],
  },
  { type: 'shot', section: 'Digits', title: 'Digits at a glance', shot: 'cnn-overview' },
  {
    type: 'text', section: 'Digits', title: 'How a CNN reads a digit',
    lead: 'A convolutional network slides small filters over the image to find strokes, then combines them into shapes and scores each digit.',
    html: `${cnnPipelineSvg()}
      <div class="columns three">
        <p><strong>Convolution</strong>: a 3 × 3 filter slides over the image, and at every position multiplies the pixels under it by its weights and adds them up. The result, a feature map, shows where the filter’s pattern appears. ReLU keeps only positive responses.</p>
        <p><strong>Max pooling</strong> keeps the strongest value of every 2 × 2 patch, halving the size so the next layer sees a wider area and small shifts matter less.</p>
        <p><strong>Dense and softmax</strong>: 64 neurons look at all 784 numbers from the last pooling, then 10 scores, one per digit, become probabilities that add up to 100%.</p>
      </div>`,
  },
  { type: 'shot', section: 'Digits', title: 'Inside the layers', shot: 'cnn-layers' },
  {
    type: 'images', section: 'Digits', title: 'The answer, and what mattered',
    images: [{ shot: 'cnn-saliency', caption: 'Left: the 28 × 28 input. Right: what mattered most.' }, { shot: 'cnn-prediction', caption: 'The prediction panel.' }],
    notes: [
      ['What mattered most', 'Gradient × input: how much each pixel’s ink changed the score of the answer. Violet ink pushed toward the answer, amber ink against it.'],
      ['The answer', 'The digit with the highest probability, and how sure the network is.'],
      ['All ten probabilities', 'Softmax output for every digit. Close calls show where the network is unsure, for example between 4 and 9.'],
      ['Fool the network', 'Pick a target digit and press Fool it (next page).'],
    ],
  },
  {
    type: 'shot', section: 'Digits', title: 'Fool the network', shot: 'cnn-fooled',
    lead: 'Networks can be tricked by changes a person would barely notice. The attack changes one pixel at a time, always the one that helps the target most, until the network reads the target.',
  },

  /* ---------- Autoencoder ---------- */
  {
    type: 'divider', section: 'Autoencoder', title: 'Autoencoder', shot: 'hero-vae',
    lead: 'A variational autoencoder learns to squeeze every handwritten digit into just two numbers and to redraw it from them. With two numbers, every digit has a place on a map, and every place on the map is a digit.',
    points: ['Drag a point across the map and watch one digit turn into another.', 'Draw your own digit to see where it lands among the others.'],
  },
  { type: 'shot', section: 'Autoencoder', title: 'The autoencoder at a glance', shot: 'vae-overview' },
  {
    type: 'text', section: 'Autoencoder', title: 'How the autoencoder works',
    lead: 'An encoder boils a digit down to two numbers; a decoder draws a digit back from any two numbers. Both were trained together on 60,000 MNIST digits.',
    html: `${autoencoderDiagram(img('vae-input'), img('vae-output'))}
      <div class="columns three">
        <p><strong>Two numbers can’t hold every detail</strong>, so the encoder keeps what tells digits apart most: which digit it is, and roughly how it is slanted and how thick.</p>
        <p><strong>The variational part</strong>: in training, the encoder gives a small cloud instead of a point, and the decoder must redraw from a random spot in it. Nearby points therefore decode to similar digits. A second term, the KL divergence, pulls every cloud toward the middle, so the map has no gaps.</p>
        <p><strong>How well it works</strong>: a vote among each test digit’s five nearest neighbors on the map names it correctly 81.6% of the time, and the Digits network reads 82% of the redrawn digits correctly. Look-alikes such as 4 and 9 overlap.</p>
      </div>`,
  },
  {
    type: 'images', section: 'Autoencoder', title: 'The decoded grid',
    lead: 'Switch the map to Decoded grid and every tile shows the digit the decoder draws at that spot.',
    images: [{ shot: 'vae-grid', caption: 'The whole map, decoded.' }],
    notes: [
      ['Smooth', 'Moving across the map, digits blend into each other: 1s lean over and become 7s, 4s become 9s.'],
      ['Regions', 'Each digit fills a region. Digits that look alike (4 and 9; 3, 5 and 8) border each other.'],
      ['Every point is a digit', 'There are no empty gaps: the KL term in training keeps the map filled.'],
    ],
  },
  { type: 'shot', section: 'Autoencoder', title: 'Where your drawing lands', shot: 'vae-example' },

  /* ---------- Recurrent ---------- */
  {
    type: 'divider', section: 'Recurrent', title: 'Recurrent', shot: 'hero-rnn',
    lead: 'A recurrent network reads text one character at a time, carrying a memory of 256 numbers from each character to the next. This one, an LSTM, was trained for this site on 19 public-domain children’s books.',
    points: ['It learned spelling and punctuation, not meaning, so its own sentences sound like English without making sense.', 'The page shows exactly what its memory holds, and compares it with a network that has no gates.'],
  },
  { type: 'shot', section: 'Recurrent', title: 'The recurrent page at a glance', shot: 'rnn-overview' },
  { type: 'shot', section: 'Recurrent', title: 'Reading the memory panel', shot: 'rnn-memory-top' },
  { type: 'shot', section: 'Recurrent', title: 'One memory cell and its gates', shot: 'rnn-memory-bottom' },
  {
    type: 'text', section: 'Recurrent', title: 'Inside an LSTM memory cell',
    lead: 'Each of the 256 memory cells is controlled by three gates: numbers between 0 and 1, recomputed at every character from the new character and the previous output.',
    html: `${lstmCellSvg()}
      <div class="columns four">
        <p><strong>Forget gate f</strong>: how much of the old memory to keep. 1 keeps it all.</p>
        <p><strong>Input gate i</strong> and <strong>candidate g</strong>: how much new information to write, and what (−1 to 1).</p>
        <p><strong>Output gate o</strong>: how much of the memory to reveal as the output h.</p>
        <p><strong>In short</strong>: new memory = f × old + i × g; output = o × tanh(memory). When f stays near 1, a cell can hold a value for dozens of characters, which is how the quotation cell remembers it is inside a quote.</p>
      </div>`,
  },
  { type: 'shot', section: 'Recurrent', title: 'Next character', shot: 'rnn-next' },
  {
    type: 'shot', section: 'Recurrent', title: 'Why gates matter', shot: 'rnn-gates-top',
    lead: 'A second recurrent network with about as many weights but no gates (512 memory numbers instead of 256) reads the same text, for comparison.',
  },
  { type: 'shot', section: 'Recurrent', title: 'Why gates matter: memory over distance', shot: 'rnn-gates-bottom' },

  /* ---------- Transformer ---------- */
  {
    type: 'divider', section: 'Transformer', title: 'Transformer', shot: 'hero-transformer',
    lead: 'GPT-2 small (OpenAI, 2019), the real 124-million-parameter language model, running in your browser. Follow a prompt through tokens, embeddings and 12 blocks of attention and MLPs, to the probability of every next word.',
    points: ['Inspired by Transformer Explainer from Georgia Tech’s Polo Club; the design and code here are independent.', 'Switch heads off to see what they actually do.'],
  },
  {
    type: 'images', section: 'Transformer', title: 'Loading the model', layout: 'stack',
    images: [{ shot: 'tf-loader', caption: 'The first visit asks before downloading.' }],
    notes: [
      ['Load the model', 'The first visit downloads GPT-2’s 130 MB of weights once; after that the browser keeps them and the model opens without asking.'],
      ['Where it runs', 'On your graphics card through WebGPU when the browser supports it, otherwise on the CPU. On an Apple M4 the GPU is about 28 to 50 times faster.'],
      ['Private', 'Your prompt never leaves the browser.'],
    ],
  },
  { type: 'shot', section: 'Transformer', title: 'Prompt and settings', shot: 'tf-dock', layout: 'stack' },
  { type: 'shot', section: 'Transformer', title: 'Three columns', shot: 'tf-overview', lead: 'Below the prompt, the model reads left to right.' },
  { type: 'shot', section: 'Transformer', title: 'Tokens and embeddings', shots: ['tf-embedding-tokens', 'tf-embedding-maps'] },
  { type: 'shot', section: 'Transformer', title: 'Blocks and heads', shot: 'tf-blocks' },
  { type: 'shot', section: 'Transformer', title: 'Inside one attention head', shot: 'tf-head' },
  {
    type: 'text', section: 'Transformer', title: 'How attention works',
    lead: 'Attention lets each token gather information from the tokens before it. Every head does this with its own learned weights, so each can track something different: the previous word, a matching name, the end of a sentence.',
    html: `${attentionSvg()}
      <div class="columns three">
        <p><strong>Compare</strong>: the token’s query is multiplied with every earlier token’s key and divided by √64. A high score means a good match.</p>
        <p><strong>Weigh</strong>: softmax turns the scores into shares between 0 and 1 that add up to 1. These are the attention weights the page draws.</p>
        <p><strong>Mix</strong>: the output is the weighted sum of the values, so well-matched tokens contribute most. 12 heads do this in parallel in each of 12 blocks: 144 heads in all.</p>
      </div>`,
  },
  {
    type: 'images', section: 'Transformer', title: 'MLP and the final state',
    images: [{ shot: 'tf-mlp', caption: 'The MLP of the open block.' }],
    notes: [
      ['MLP', 'After attention, each token’s vector goes through a two-layer network on its own: 768 → 3,072 neurons with GELU → 768. One row per token, one column per neuron; violet positive, amber negative.'],
      ['Residual stream', 'Each block adds its attention and MLP results to the vector it received, so information builds up block by block.'],
      ['Final hidden state', 'After the 12th block and a final normalization, the last token’s 768 numbers are scored against all 50,257 token embeddings. Those scores become the next-token probabilities.'],
    ],
  },
  { type: 'shot', section: 'Transformer', title: 'Choosing the next token', shot: 'tf-next' },
  {
    type: 'text', section: 'Transformer', title: 'Temperature, top-k and top-p',
    lead: 'GPT-2 gives every one of its 50,257 tokens a score. Softmax turns scores into probabilities; temperature divides the scores first. A filter then decides which tokens Generate may pick, and one is drawn at random.',
    html: `${samplingSvg()}
      <div class="columns three">
        <p><strong>Temperature</strong> below 1 sharpens the probabilities toward the likeliest token: safer, more repetitive text. Above 1 flattens them: more surprising text, more mistakes.</p>
        <p><strong>Top-k</strong> keeps only the k likeliest tokens (5 by default) and shares the probability among them.</p>
        <p><strong>Top-p</strong> keeps the fewest tokens whose probabilities add up to p, so it keeps few when the model is sure and many when it isn’t.</p>
      </div>
      <p class="footnote">Illustrative scores for six tokens; the page computes this over all 50,257.</p>`,
  },
  {
    type: 'shot', section: 'Transformer', title: 'How the guess forms', shot: 'tf-lens',
    lead: 'The logit lens reads the model’s mind midway: after the embedding and after each block, the hidden state is scored against the vocabulary as if the model stopped there.',
  },
  {
    type: 'shot', section: 'Transformer', title: 'Surprising words', shot: 'tf-surprise', layout: 'stack',
    lead: 'Surprise is measured in bits: −log₂ of the probability GPT-2 gave a token before seeing it. 1 bit was a 50% guess; 10 bits about 1 in 1,000.',
  },
  {
    type: 'shot', section: 'Transformer', title: 'Switching off a head', shot: 'tf-off',
    lead: 'Switch off a head to see what it was contributing. Here block 12, head 11 is off.',
  },
  {
    type: 'shot', section: 'Transformer', title: 'What switching it off changed', shot: 'tf-off-next',
    after: 'A real finding: block 12 head 11 and block 11 head 8 hold “Mary” back; switching them off raises it. They match the “negative name mover” heads that interpretability research found in GPT-2 small (Wang et al., 2022). A line under the prompt counts the heads switched off, with Turn all back on.',
  },
  { type: 'shot', section: 'Transformer', title: 'Which heads matter', shot: 'tf-importance' },

  /* ---------- Site-wide ---------- */
  {
    type: 'images', section: 'Across the site', title: 'Light and dark', layout: 'stack',
    images: [{ shot: 'theme-light', caption: 'Light theme.' }, { shot: 'theme-dark', caption: 'Dark theme.' }],
    notes: [
      ['Same meaning', 'Each theme has its own violet and amber, checked for contrast and for color blindness, so the colors mean the same in both.'],
      ['Nothing resets', 'Switching theme repaints every chart and keeps your experiment, text or drawing.'],
    ],
  },
  {
    type: 'images', section: 'Across the site', title: 'On a phone',
    images: [{ shot: 'phone-playground', caption: 'Playground' }, { shot: 'phone-vae', caption: 'Autoencoder' }, { shot: 'phone-rnn', caption: 'Recurrent' }],
    notes: [
      ['Touch', 'Draw, drag and tap with a finger: the scratchpads, the map point and every control work by touch.'],
      ['Fits', 'Every page fits screens from 320 pixels wide; wide charts scroll sideways inside their panel.'],
    ],
  },
  {
    type: 'text', section: 'Across the site', title: 'Under the hood',
    html: `<div class="columns two roomy">
      <div><h3>Everything runs in your browser</h3><p>Training, drawing, reading and generating all happen on your device. There is no server, no account, and nothing you type or draw is uploaded. The models are files that come with the page; GPT-2 downloads once and is then kept by the browser.</p></div>
      <div><h3>Fast where it can be</h3><p>GPT-2 runs on your graphics card through WebGPU when the browser supports it, and checks itself against the CPU before it is trusted. Training in the playground runs in a background worker so the page stays smooth.</p></div>
      <div><h3>For everyone</h3><p>Every control works from the keyboard, with visible focus; charts have text descriptions for screen readers; and if your system asks for reduced motion, animations are switched off.</p></div>
      <div><h3>Tested</h3><p>95 unit tests check the maths (gradients against finite differences, GPU against CPU) and 50 browser tests drive every page, on every change.</p></div>
    </div>`,
  },
  {
    type: 'text', section: 'Across the site', title: 'Glossary',
    html: `<dl class="glossary">
      <div><dt>Activation function</dt><dd>The bend a neuron applies to its weighted sum, such as tanh or ReLU.</dd></div>
      <div><dt>Attention</dt><dd>How much one token draws on each earlier token, as shares adding up to 1.</dd></div>
      <div><dt>Backpropagation</dt><dd>Working out every weight’s gradient by passing the error backward through the network.</dd></div>
      <div><dt>Batch</dt><dd>The group of examples averaged into one weight update.</dd></div>
      <div><dt>Bias</dt><dd>A number added to a neuron’s sum, letting it shift its threshold.</dd></div>
      <div><dt>Bit</dt><dd>A unit of surprise: each extra bit means half as likely.</dd></div>
      <div><dt>Decoder, encoder</dt><dd>The halves of an autoencoder: one compresses, the other rebuilds.</dd></div>
      <div><dt>Embedding</dt><dd>A list of learned numbers standing for a token or position.</dd></div>
      <div><dt>Epoch</dt><dd>One pass over all the training data.</dd></div>
      <div><dt>Feature map</dt><dd>Where one convolution filter found its pattern in an image.</dd></div>
      <div><dt>Gate</dt><dd>In an LSTM, a number from 0 to 1 that decides how much to keep, write or reveal.</dd></div>
      <div><dt>Gradient</dt><dd>Which way, and how strongly, changing a number changes the loss.</dd></div>
      <div><dt>Head</dt><dd>One attention mechanism with its own weights; GPT-2 small has 144.</dd></div>
      <div><dt>Latent space</dt><dd>The space of compressed codes; here, the autoencoder’s two-number map.</dd></div>
      <div><dt>Learning rate</dt><dd>The size of each training step.</dd></div>
      <div><dt>Logit</dt><dd>A raw score before softmax turns scores into probabilities.</dd></div>
      <div><dt>Loss</dt><dd>One number measuring how wrong the network is; training lowers it.</dd></div>
      <div><dt>Neuron</dt><dd>Adds up weighted inputs and a bias, then applies an activation.</dd></div>
      <div><dt>Overfitting</dt><dd>Memorizing the training data instead of learning the pattern.</dd></div>
      <div><dt>Parameter</dt><dd>A learned number: a weight or a bias.</dd></div>
      <div><dt>Regularization</dt><dd>A penalty that keeps weights small, against overfitting.</dd></div>
      <div><dt>Softmax</dt><dd>Turns scores into probabilities that add up to 1.</dd></div>
      <div><dt>Temperature</dt><dd>Sharpens (below 1) or flattens (above 1) probabilities before sampling.</dd></div>
      <div><dt>Token</dt><dd>A piece of text GPT-2 reads as one unit: a word, part of a word or a symbol.</dd></div>
      <div><dt>Weight</dt><dd>The strength of a connection; positive weights excite, negative ones inhibit.</dd></div>
    </dl>`,
  },
  {
    type: 'text', section: 'Across the site', title: 'Credits',
    html: `<div class="credits-list">
      <p><strong>TensorFlow Playground</strong> by Daniel Smilkov and Shan Carter (Google, Apache 2.0). The Playground is an adaptation of it.</p>
      <p><strong>GPT-2 small</strong> by OpenAI (MIT license), with its original tokenizer; the weights are converted to 8-bit blocks for the browser.</p>
      <p><strong>MNIST</strong> by Yann LeCun, Corinna Cortes and Christopher J. C. Burges (CC BY-SA 3.0): training data for Digits and Autoencoder.</p>
      <p><strong>Project Gutenberg</strong>: 19 public-domain children’s books that trained the Recurrent page’s networks.</p>
      <p><strong>Transformer Explainer</strong> by the Polo Club at Georgia Tech: the inspiration for the Transformer page.</p>
      <p><strong>Wang et al. (2022)</strong>, “Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small”: the name-mover heads.</p>
      <p><strong>Mona Sans</strong> by GitHub (SIL Open Font License): the typeface.</p>
      <p>Every model on the Digits, Autoencoder and Recurrent pages was trained for this site, with reproducible scripts in the project.</p>
    </div>`,
  },
];
