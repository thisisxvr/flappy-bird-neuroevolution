namespace Network {
  export class NeuralNetwork {
    public fitness: number
    private _weights: number[]
    private layers: Layer[]

    constructor([inputNodeCount, ...otherLayers]: Array<number | number[]>) {
      const outputNodeCount = otherLayers.pop()
      const hiddenNodeCount = otherLayers[0] as number[]
      this.layers = [], this._weights = []
      let nodesInPreviousLayer = 0, index = 0
      // Input layer.
      this.layers[0] = new Layer(inputNodeCount as number, nodesInPreviousLayer, index)
      nodesInPreviousLayer = inputNodeCount as number
      // Hidden layers.
      for (const i in hiddenNodeCount as number[]) {
        const nodeCount = hiddenNodeCount[i]
        this.layers.push(new Layer(nodeCount, nodesInPreviousLayer, ++index))
        nodesInPreviousLayer = hiddenNodeCount[i]
      }
      // Output layer.
      this.layers.push(new Layer(outputNodeCount as number, nodesInPreviousLayer, ++index))
    }

    // Returns a flat array with weights of ALL neurons in the network.
    get weights() {
      if (this._weights.length) { return this._weights }
      for (const layer of this.layers) {
        for (const neuron of layer) { this._weights.concat(neuron.weights) }
      }
      return this._weights
    }

    // Persists the mutated weights to the neurons.
    persist() {
      // tslint:disable-next-line:prefer-for-of
      let index = 0
      for (const layer of this.layers) {
        for (const neuron of layer) {
          for (const w in neuron.weights) {
            neuron.weights[w] = this._weights[index]
            index++
          }
        }
      }
    }

    // Computes the output of the network.
    compute(inputs: number[]): number[] {
      // Pass inputs to the input layer.
      const inputLayer = this.layers[0]
      for (const i in inputs) {
        const inputNeuron = inputLayer.nodes[i]
        inputNeuron.value = inputs[i]
      }

      // Compute outputs of hidden layers and output layer.
      let previousLayer = inputLayer
      for (let i = 1; i < this.layers.length; i++) {
        const layer = this.layers[i]
        for (const neuron of layer) {
          for (const previousNeuron of previousLayer) {
            let sum = 0
            for (const weight of neuron.weights) { sum += previousNeuron.value * weight }
            neuron.activate(sum)
          }
        }
        previousLayer = layer
      }

      // Finally, get the output value(s).
      const outputs = []
      const outputLayer = this.layers[this.layers.length - 1]
      for (const outputNeuron of outputLayer) { outputs.push(outputNeuron.value) }

      return outputs
    }
  }

  class Layer {
    private id: number
    private _nodes: Neuron[]
    private pointer = 0
    get nodes() { return this._nodes }

    constructor(nodeCount: number, inputCount: number, index = 0) {
      this.id = index
      this._nodes = new Array<Neuron>(nodeCount)
      for (let i = 0; i < nodeCount; i++) { this._nodes[i] = new Neuron(inputCount) }
    }

    // Returns a Neuron when iterating over a Layer object.
    next(): IteratorResult<Neuron> {
      if (this.pointer < this._nodes.length) {
        return { done: false, value: this._nodes[this.pointer++] }
      }
      this.pointer = 0
      return { done: true, value: undefined! }
    }

    // Allows for iterating over a Layer object.
    [Symbol.iterator](): IterableIterator<Neuron> { return this }
  }

  class Neuron {
    public value: number
    public weights: number[]

    constructor(weightCount: number) {
      this.value = undefined!
      this.weights = new Array(weightCount)
      for (let i = 0; i < weightCount; i++) { this.weights[i] = Math.random() * 2 - 1 }
    }

    // Sigmoid activation function.
    activate(sum: number): number {
      const theta = -sum / 1
      return this.value = 1 / (1 + Math.exp(theta))
    }
  }
}