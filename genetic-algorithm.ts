/// <reference path="neural-network.ts" />

namespace GeneticAlgorithm {
  import NeuralNetwork = Network.NeuralNetwork
  const crossoverRate  = 0.5
  const elitismRate    = 0.2
  const mutationRate   = 0.1
  const mutationRange  = 0.5
  const randomnessRate = 0.2
  const generations: Generation[] = []
  let populationCount: number
  export function evolve(networkShape = [2, [2], 1], popCount = 50): Generation {
    populationCount = popCount
    if (generations.length === 0) {
      const firstGeneration = new Generation(networkShape, populationCount)
      generations.push(firstGeneration)
      return firstGeneration
    } else {
      const currentGeneration = generations[generations.length - 1]
      const nextGeneration = currentGeneration.next()
      generations.push(nextGeneration)
      return nextGeneration
    }
  }

  export class Generation {
    private _population: NeuralNetwork[]
    get Population() { return this._population }

    constructor(networkShape = [2, [2], 1], popCount = populationCount, newPopulation?: NeuralNetwork[]) {
      if (newPopulation) {
        this._population = this.rank(newPopulation)
        return this
      }

      this._population = new Array<NeuralNetwork>(popCount)

      for (let i = 0; i < popCount; i++) {
        this._population[i] = new NeuralNetwork(networkShape)
      }
    }

    // Sorts the population in descending order, ranked by score.
    private rank(population = this._population): NeuralNetwork[] {
      const mapped = population.map((chromosome, index) => {
        return { index, score: chromosome.fitness }
      })

      mapped.sort((a, b) => {
        if (a.score > b.score) { return 1 }
        if (a.score < b.score) { return -1 }
        return 0
      })

      return mapped.map((chromosome) => { return population[chromosome.index] })
    }

    // Takes a pair of parents, performs crossover and mutation and returns the offspring.
    private breed(parentOne: NeuralNetwork, parentTwo: NeuralNetwork, numberOfOffspring = 1): NeuralNetwork[] {
      function crossover(parentOne: NeuralNetwork, parentTwo: NeuralNetwork): NeuralNetwork {
        const child = parentOne

        for (const i in parentTwo.Weights) {
          if (Math.random() <= crossoverRate) { child.Weights[i] = parentTwo.Weights[i] }
        }
        return child
      }

      function mutate(chromosome: NeuralNetwork) {
        for (const i in chromosome.Weights) {
          if (Math.random() <= mutationRate) {
            chromosome.Weights[i] += Math.random() * mutationRange * 2 - mutationRange
          }
        }
      }

      const offspring = new Array<NeuralNetwork>(numberOfOffspring)

      for (let i = 0; i < numberOfOffspring; i++) {
        const child = crossover(parentOne, parentTwo)
        mutate(child)
        child.persist()
        offspring[i] = child
      }

      return offspring
    }

    // Returns the next generation.
    next(): Generation {
      const nextGen = []
      this.rank()

      // Push some of the elite chromosomes into the next generation unchanged.
      for (let i = 0; i < Math.round(elitismRate * populationCount); i++) {
        if (nextGen.length < populationCount) { nextGen.push(this._population[i]) }
      }

      // Introduce some randomness into the next generation.
      for (let i = 0; i < Math.round(randomnessRate * populationCount); i++) {
        const firstNetwork = this._population[0]
        for (const i in firstNetwork.Weights) {
          if (nextGen.length < populationCount) { nextGen.push(firstNetwork.Weights[i] = this.randomClamped()) }
        }
      }

      // Breed new children from the elites until we hit the population limit.
      // let ceiling = 1
      for (let ceiling = 1; ceiling < populationCount; ceiling++) {
        for (let i = 0; i < ceiling; i++) {
          const children = this.breed(this._population[i], this._population[ceiling])
          for (const child of children) { nextGen.push(child) }
          if (nextGen.length >= populationCount) { break }
        }
        if (nextGen.length >= populationCount) { break }
        // ceiling++
        // if (ceiling >= this._population.length - 1) { ceiling = 0 }
      }

      return new Generation(undefined, undefined, nextGen as NeuralNetwork[])
    }

    // Returns a random value between -1 and 1.
    private randomClamped(): number { return Math.random() * 2 - 1 }
  }
}