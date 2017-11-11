
class NeuralNetwork {
  private _fitness: number
  get Fitness() { return this._fitness }
  private _weights: number[]
  get Weights() { return this._weights }
  private layers: Layer[]

  constructor([input, hidden, output]: Array<(number | number[])>) {
    this.layers = []
    let nodesInPreviousLayer = 0, index = 0
    // Input layer.
    this.layers[0] = new Layer(input as number, nodesInPreviousLayer, index)
    nodesInPreviousLayer = input as number
    // Hidden layers.
    for (const i in hidden as number[]) {
      const nodeCount = hidden[i] as number
      this.layers[1][i] = new Layer(nodeCount, nodesInPreviousLayer, ++index)
      nodesInPreviousLayer = hidden[i]
    }
    // Output layer.
    this.layers[2] = new Layer(output as number, nodesInPreviousLayer, ++index)
  }

  // Computes the output of the network.
  compute(inputs: number[]): number {
    // Pass inputs to the input layer.
    const inputLayer = this.layers[0]
    for (const i in inputs) {
      for (const inputNode of inputLayer) {
        inputNode.value = inputs[i]
      }
    }

    // Compute outputs of hidden layers.
    let previousLayer = inputLayer
    for (let i = 1; i < this.layers.length; i++) {
      const hiddenLayer = this.layers[i]
      for (const node of hiddenLayer) {
        for (const previousNode of previousLayer) {
          node.activate(previousNode.value)
        }
      }
      previousLayer = hiddenLayer
    }

    // Finally, get the output value.
    let output = 0
    const outputLayer = this.layers[this.layers.length - 1]
    for (const node of outputLayer) {
      for (const previousNode of previousLayer) {
        output += node.activate(previousNode.value)
      }
    }

    return output
  }
}

class Layer {
  private id: number
  private nodes: Neuron[]
  private pointer = 0

  constructor(nodeCount: number, inputCount: number, index = 0) {
    this.id = index
    this.nodes = new Array<Neuron>(nodeCount)
    for (let i = 0; i < nodeCount; i++) { this.nodes[i] = new Neuron(inputCount) }
  }

  next(): IteratorResult<Neuron> {
    if (this.pointer < this.nodes.length) {
      return {
        done: false,
        value: this.nodes[this.pointer++]
      }
    }
    return {
      done: true,
      // tslint:disable-next-line:no-null-keyword
      value: null!
    }
  }

  [Symbol.iterator](): IterableIterator<Neuron> { return this }
}

class Neuron {
  public value: number
  public weights: number[]

  constructor(n: number) {
    this.value = 0
    this.weights = new Array(n)
    for (let i = 0; i < n; i++) { this.weights[i] = Math.random() * 2 - 1 }
  }

  // Sigmoid activation function.
  // Computes the output of the neuron.
  activate(sum = 0): number {
    const theta = -sum / 1
    return 1 / (1 + Math.exp(theta))
  }
}

export default NeuralNetwork