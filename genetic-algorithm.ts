import NeuralNetwork from "./neural-network"

class GeneticAlgorithm {
  private population: NeuralNetwork[]
  private mutationRate = 0.1
  private mutationRange = 0.5
  private crossoverProbability = 0.5

  constructor(populationCount: number) {
    this.population = new Array<NeuralNetwork>(populationCount)
  }

  evolve() {

  }

  breed(parentOne: NeuralNetwork, parentTwo: NeuralNetwork, numberOfOffspring: number): NeuralNetwork[] {
    let offspring = new Array<NeuralNetwork>(numberOfOffspring)

    for (let i = 0; i < numberOfOffspring; i++) {
      let child = this.crossover(parentOne, parentTwo)
      this.mutate(child)
      offspring[i] = child
    }

    return offspring
  }

  private crossover(parentOne: NeuralNetwork, parentTwo: NeuralNetwork): NeuralNetwork {
    let child = parentOne

    for (let i in parentTwo.weights) {
      if (Math.random() <= this.crossoverProbability) {
        child.weights[i] = parentTwo.weights[i]
      }
    }

    return child
  }

  private mutate(chromosome: NeuralNetwork) {
    for (let i in chromosome.weights) {
      if (Math.random() <= this.mutationRate) {
        chromosome.weights[i] += Math.random() * this.mutationRange * 2 - this.mutationRange
      }
    }
  }

}

export default GeneticAlgorithm