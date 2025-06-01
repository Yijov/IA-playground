export class PolicyNetwork {
  layers: number[];
  weights: number[][][];
  biases: number[][];

  constructor(layers: number[]) {
    this.layers = layers;
    this.weights = [];
    this.biases = [];
    this.initializeWeights();
  }

  private initializeWeights() {
    for (let i = 1; i < this.layers.length; i++) {
      const layerWeights = [];
      const layerBiases = [];

      for (let j = 0; j < this.layers[i]; j++) {
        layerBiases.push(this.randomWeight());
        const neuronWeights = [];
        for (let k = 0; k < this.layers[i - 1]; k++) {
          neuronWeights.push(this.randomWeight());
        }
        layerWeights.push(neuronWeights);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  private softmax(xs: number[]): number[] {
    const max = Math.max(...xs);
    const exps = xs.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
  }

  private randomWeight(): number {
    return Math.random() * 2 - 1;
  }

  forward(input: number[]): { activations: number[][]; zs: number[][] } {
    let activations = [input];
    let zs: number[][] = [];

    for (let l = 0; l < this.weights.length; l++) {
      const weight = this.weights[l];
      const bias = this.biases[l];
      const prevActivation = activations[l];

      const z = weight.map(
        (neuronWeights, j) =>
          neuronWeights.reduce((sum, w, k) => sum + w * prevActivation[k], 0) +
          bias[j]
      );

      const activation =
        l === this.weights.length - 1
          ? this.softmax(z) // Output layer
          : z.map(this.relu);

      zs.push(z);
      activations.push(activation);
    }

    return { activations, zs };
  }

  predict(input: number[]): number[] {
    return this.forward(input).activations.at(-1)!;
  }

  sampleAction(input: number[]): number {
    const probs = this.predict(input);
    let r = Math.random();
    for (let i = 0; i < probs.length; i++) {
      r -= probs[i];
      if (r <= 0) return i;
    }
    return probs.length - 1;
  }

  trainEpisode(
    states: number[][],
    actions: number[],
    discountedRewards: number[],
    learningRate: number
  ) {
    const gradsW = this.weights.map(layer => layer.map(neuron => neuron.map(() => 0)));
    const gradsB = this.biases.map(layer => layer.map(() => 0));

    for (let t = 0; t < states.length; t++) {
      const { activations, zs } = this.forward(states[t]);
      const output = activations.at(-1)!;
      const action = actions[t];
      const reward = discountedRewards[t];

      const delta = output.map((p, i) => (i === action ? p - 1 : p));
      for (let i = 0; i < delta.length; i++) {
        delta[i] *= reward;
      }

      const deltas: number[][] = [delta];

      for (let l = this.weights.length - 2; l >= 0; l--) {
        const z = zs[l];
        const sp = z.map(this.reluDerivative);
        const nextDelta = deltas[0];
        const currentWeights = this.weights[l + 1];
        const delta = [];

        for (let j = 0; j < z.length; j++) {
          let error = 0;
          for (let k = 0; k < nextDelta.length; k++) {
            error += currentWeights[k][j] * nextDelta[k];
          }
          delta.push(error * sp[j]);
        }

        deltas.unshift(delta);
      }

      for (let l = 0; l < this.weights.length; l++) {
        const delta = deltas[l];
        const aPrev = activations[l];

        for (let j = 0; j < this.weights[l].length; j++) {
          for (let k = 0; k < this.weights[l][j].length; k++) {
            gradsW[l][j][k] += delta[j] * aPrev[k];
          }
          gradsB[l][j] += delta[j];
        }
      }
    }

    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        for (let k = 0; k < this.weights[l][j].length; k++) {
          this.weights[l][j][k] -= (learningRate / states.length) * gradsW[l][j][k];
        }
        this.biases[l][j] -= (learningRate / states.length) * gradsB[l][j];
      }
    }
  }

  save(key: string = "policy-network") {
    const model = {
      layers: this.layers,
      weights: this.weights,
      biases: this.biases,
    };
    localStorage.setItem(key, JSON.stringify(model));
  }

  static load(key: string = "policy-network"): PolicyNetwork | null {
    const modelStr = localStorage.getItem(key);
    if (!modelStr) return null;

    const model = JSON.parse(modelStr);
    const nn = new PolicyNetwork(model.layers);
    nn.weights = model.weights;
    nn.biases = model.biases;
    return nn;
  }
}
