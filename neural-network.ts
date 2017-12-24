namespace Network {

  // Data structure to hold network information for the algorithm to operate on.
  export interface INetworkData {
    neurons: number[], // Number of neurons per layer
    weights: number[]  // and their corresponding weights.
  }

  export class NeuralNetwork {
    public fitness: number
    private _weights: number[]
    private layers: Layer[]

    constructor([inputNodeCount, ...otherLayers]: Array<number | number[]> = [2, [2], 1]) {
      const outputNodeCount  = otherLayers.pop() as number
      const hiddenLayerCount = otherLayers[0] as number[]
      this.layers = [], this._weights = []
      let nodesInPreviousLayer = 0

      // Input layer.
      this.layers[0]       = new Layer(inputNodeCount as number, nodesInPreviousLayer)
      nodesInPreviousLayer = inputNodeCount as number

      // Hidden layers.
      for (const i in hiddenLayerCount as number[]) {
        const nodeCount = hiddenLayerCount[i]
        this.layers.push(new Layer(nodeCount, nodesInPreviousLayer))
        nodesInPreviousLayer = hiddenLayerCount[i]
      }

      // Output layer.
      this.layers.push(new Layer(outputNodeCount as number, nodesInPreviousLayer))
    }

    // Returns the network topology
    // and a flat array with weights of all the neurons.
    get weights() {
      const data: INetworkData = { neurons: [], weights: [] }

      // if (this._weights.length > 0) { return this._weights }
      for (const layer of this.layers) {
        data.neurons.push(layer.nodes.length)
        for (const node of layer) {
          this._weights = this._weights.concat(node.weights)
        }
      }

      data.weights = this._weights
      return data
    }

    /** Persists the mutated weights to the neurons. */
    persist(data: INetworkData) {
      let nodesInPreviousLayer = 0
      let index                = 0
      this.layers              = []

      for (const i in data.neurons) {
        const layer = new Layer(data.neurons[i], nodesInPreviousLayer)
        for (const node of layer) {
          for (const k in node.weights) {
            node.weights[k] = data.weights[index++]
          }
        }
        nodesInPreviousLayer = data.neurons[i]
        this.layers.push(layer)
      }
    }

    /** Computes the output of the network. */
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

        for (const node of layer) {
          let sum = 0
          for (const k in previousLayer.nodes) {
            sum += previousLayer.nodes[k].value * node.weights[k]
          }
          node.activate(sum)
        }

        previousLayer = layer
      }

      // Finally, get the output value(s).
      const outputs     = []
      const outputLayer = this.layers[this.layers.length - 1]
      for (const node of outputLayer) { outputs.push(node.value) }
      return outputs
    }
  }

  class Layer {
    private _nodes: Neuron[]
    private pointer = 0
    get nodes() { return this._nodes }

    constructor(nodeCount: number, inputCount: number) {
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
      this.value   = undefined!
      this.weights = new Array(weightCount)
      for (let i = 0; i < weightCount; i++) { this.weights[i] = this.randomClamped() }
    }

    /** Returns a random value between -1 and 1. */
    private randomClamped(): number { return Math.random() * 2 - 1 }

    /**
     * Logistic activation function.
     * Calculates the neuron's output.
     */
    activate(sum: number): number {
      const theta = (-sum) / 1
      return this.value = (1 / (1 + Math.exp(theta)))
    }
  }

}