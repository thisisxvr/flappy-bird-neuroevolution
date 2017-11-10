import NeuralNetwork from "./neural-network"

namespace GeneticAlgorithm {

  const mutationRate = 0.1
  const mutationRange = 0.5
  const crossoverRate = 0.5
  let generations: Generation[]

  function evolve() {
    if (generations.length === 0) {
      //
    }
  }

  class Generation {
    private population: NeuralNetwork[]

    constructor(populationCount: number) {
      for (let i = 0; i < populationCount; i++) {
        this.population[i] = new NeuralNetwork()
      }
    }

    sortPopulation() {
      const mapped = this.population.map((chromosome, index) => {
        return { index, score: chromosome.fitness }
      })

      mapped.sort((a, b) => {
        if (a.score > b.score) { return 1 }
        if (a.score < b.score) { return -1 }
        return 0
      })

      const temp = mapped.map((chromosome) => { return this.population[chromosome.index] })
      this.population = temp
    }

    breed(parentOne: NeuralNetwork, parentTwo: NeuralNetwork, numberOfOffspring: number): NeuralNetwork[] {
      function crossover(parentOne: NeuralNetwork, parentTwo: NeuralNetwork): NeuralNetwork {
        const child = parentOne

        for (const i in parentTwo.weights) {
          if (Math.random() <= crossoverRate) { child.weights[i] = parentTwo.weights[i] }
        }

        return child
      }

      function mutate(chromosome: NeuralNetwork) {
        for (const i in chromosome.weights) {
          if (Math.random() <= mutationRate) {
            chromosome.weights[i] += Math.random() * mutationRange * 2 - mutationRange
          }
        }
      }

      const offspring = new Array<NeuralNetwork>(numberOfOffspring)

      for (let i = 0; i < numberOfOffspring; i++) {
        const child = crossover(parentOne, parentTwo)
        mutate(child)
        offspring[i] = child
      }

      return offspring
    }

  }
}
export default GeneticAlgorithm