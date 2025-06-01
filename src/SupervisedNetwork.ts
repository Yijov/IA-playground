export class SupervisedNetwork {
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

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }

  private randomWeight(): number {
    return Math.random() * 2 - 1; // [-1, 1]
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
      const activation = z.map((x) =>
        l === this.weights.length - 1 ? this.sigmoid(x) : this.relu(x)
      );

      zs.push(z);
      activations.push(activation);
    }

    return { activations, zs };
  }

  predict(input: number[]): number[] {
    return this.forward(input).activations.at(-1)!;
  }

  train(
    inputs: number[][],
    targets: number[][],
    epochs: number,
    learningRate: number
  ) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < inputs.length; i++) {
        const x = inputs[i];
        const y = targets[i];

        const { activations, zs } = this.forward(x);

        const deltas: number[][] = [];
        const output = activations.at(-1)!;
        const outputZ = zs.at(-1)!;
        const outputDelta = output.map(
          (o, j) => (o - y[j]) * this.sigmoidDerivative(outputZ[j])
        );
        deltas.unshift(outputDelta);

        for (let l = this.weights.length - 2; l >= 0; l--) {
          const z = zs[l];
          const sp = z.map(this.reluDerivative.bind(this));
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
          const layer = this.weights[l];
          const delta = deltas[l];
          const prevActivation = activations[l];

          for (let j = 0; j < layer.length; j++) {
            for (let k = 0; k < layer[j].length; k++) {
              layer[j][k] -= learningRate * delta[j] * prevActivation[k];
            }
            this.biases[l][j] -= learningRate * delta[j];
          }
        }
      }
    }
  }

  save(key: string = "neural-network") {
    const model = {
      layers: this.layers,
      weights: this.weights,
      biases: this.biases,
    };
    localStorage.setItem(key, JSON.stringify(model));
  }

  static load(key: string = "neural-network"): SupervisedNetwork | null {
    const modelStr = localStorage.getItem(key);
    if (!modelStr) return null;

    const model = JSON.parse(modelStr);
    const nn = new SupervisedNetwork(model.layers);
    nn.weights = model.weights;
    nn.biases = model.biases;
    return nn;
  }
}