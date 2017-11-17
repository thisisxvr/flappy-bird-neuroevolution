namespace Network {
  interface IData {
    neurons: number[], // Number of Neurons per layer.
    weights: number[]  // Weights of each Neuron's inputs.
  }

  export class NeuralNetwork {
    public fitness: number
    private _weights: number[]
    private layers: Layer[]
    private _inputNodeCount: number
    private _hiddenLayerCount: number[]
    private _outputNodeCount: number

    constructor([inputNodeCount, ...otherLayers]: Array<number | number[]> = [1, [2], 1]) {
      this._inputNodeCount = inputNodeCount as number
      this._outputNodeCount = otherLayers.pop() as number
      this._hiddenLayerCount = otherLayers[0] as number[]
      this.layers = [], this._weights = []
      let nodesInPreviousLayer = 0, index = 0
      // Input layer.
      this.layers[0] = new Layer(inputNodeCount as number, nodesInPreviousLayer, index)
      nodesInPreviousLayer = inputNodeCount as number
      // Hidden layers.
      for (const i in this._hiddenLayerCount as number[]) {
        const nodeCount = this._hiddenLayerCount[i]
        this.layers.push(new Layer(nodeCount, nodesInPreviousLayer, ++index))
        nodesInPreviousLayer = this._hiddenLayerCount[i]
      }
      // Output layer.
      this.layers.push(new Layer(this._outputNodeCount as number, nodesInPreviousLayer, ++index))
    }

    // Returns a flat array with weights of ALL neurons in the network.
    get weights() {
      if (this._weights.length > 0) { return this._weights }
      for (const layer of this.layers) {
        for (const i in layer.nodes) {
          this._weights = this._weights.concat(layer.nodes[i].weights)
        }
      }
      return this._weights
    }

    getSave() {
      const datas: IData = { neurons: [], weights: [] }

      // tslint:disable-next-line:no-invalid-this
      for (const i in this.layers) {
        datas.neurons.push(this.layers[i].nodes.length)
        for (const j in this.layers[i].nodes) {
          for (const k in this.layers[i].nodes[j].weights) {
              // push all input weights of each Neuron of each Layer into a flat
              // array.
            datas.weights.push(this.layers[i].nodes[j].weights[k])
          }
        }
      }
      return datas
    }

    setSave(save: IData) {
      let previousNeurons = 0
      let index = 0
      let indexWeights = 0
      this.layers = []
      for (const i in save.neurons) {
        // Create and populate layers.
        const layer = new Layer(save.neurons[i], previousNeurons, index)
        for (const j in layer.nodes) {
          for (const k in layer.nodes[j].weights) {
              // Apply neurons weights to each Neuron.
            layer.nodes[j].weights[k] = save.weights[indexWeights]

            indexWeights++ // Increment index of flat array.
          }
        }
        previousNeurons = save.neurons[i]
        index++
        this.layers.push(layer)
      }
    }

    // Persists the mutated weights to the neurons.
    persist() {
      // tslint:disable-next-line:prefer-for-of
      let index = 0
      // for (const layer of this.layers) {
      //   const nodeCount = layer.nodes.length
      //   const
      //   for (const i in layer.nodes) {
      //     const node = layer.nodes[i]
      //     node.value = undefined!
      //     for (const j in node.weights) {
      //       node.weights[j] = this._weights[index]
      //       index++
      //     }
      //   }
      // }
      let nodesInPreviousLayer = 0, layerID = 0
      // Input layer.
      this.layers[0] = new Layer(this._inputNodeCount as number, nodesInPreviousLayer, layerID)
      nodesInPreviousLayer = this._inputNodeCount as number
      // Hidden layers.
      for (const i in this._hiddenLayerCount as number[]) {
        const nodeCount = this._hiddenLayerCount[i]
        const layer = new Layer(nodeCount, nodesInPreviousLayer, ++layerID)
        for (const j in layer.nodes) {
          const node = layer.nodes[j]
          for (const k in node.weights) {
            node.weights[k] = this._weights[index]
            index++
          }
        }
        this.layers[1] = layer
        nodesInPreviousLayer = this._hiddenLayerCount[i]
      }
      // Output layer.
      const layer = new Layer(this._outputNodeCount as number, nodesInPreviousLayer, ++layerID)
      for (const i in layer.nodes) {
        const node = layer.nodes[i]
        for (const j in node.weights) {
          node.weights[j] = this._weights[index]
          index++
        }
      }
      this.layers[2] = layer
      this._weights = []
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
        // for (const neuron of layer) {
        //   for (const previousNeuron of previousLayer) {
        //     let sum = 0
        //     for (const weight of neuron.weights) { sum += previousNeuron.value * weight }
        //     neuron.activate(sum)
        //   }
        // }
        for (const j in layer.nodes) {
          let sum = 0
          for (const k in previousLayer.nodes) {
            sum += previousLayer.nodes[k].value * layer.nodes[j].weights[k]
          }
          layer.nodes[j].activate(sum)
        }
        previousLayer = layer
      }

      // Finally, get the output value(s).
      const outputs = []
      const outputLayer = this.layers[this.layers.length - 1]
      for (const i in outputLayer.nodes) { outputs.push(outputLayer.nodes[i].value) }
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
    // [Symbol.iterator](): IterableIterator<Neuron> { return this }
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
      const theta = (-sum) / 1
      return this.value = (1 / (1 + Math.exp(theta)))
    }
  }
}