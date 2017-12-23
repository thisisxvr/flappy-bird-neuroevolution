/// <reference path="neural-network.ts" />

namespace GeneticAlgorithm {
  import IData = Network.IData
  import NeuralNetwork = Network.NeuralNetwork
  const elitismRate    = 0.2    // Best networks pushed unchanged into the next gen.
  const randomnessRate = 0.2    // New random networks for the next generation.
  const crossoverRange = 0.5    // Uniform crossover range.
  const mutationRate   = 0.1    // Mutation rate on weights.
  const mutationRange  = 0.5    // Interval of the mutation changes on the genome's weight.
  const generations:     NeuralNetwork[][] = []
  let populationCount: number

  export function evolve(networkShape = [2, [2], 1], popCount = 50): NeuralNetwork[] {
    populationCount = popCount
    if (generations.length === 0) {
      const firstGeneration = []
      for (let i = 0; i < popCount; i++) {
        const nn = new NeuralNetwork(networkShape)
        firstGeneration.push(nn)
      }
      generations.push(firstGeneration)
      return firstGeneration
    } else {
      // const currentGeneration = generations[generations.length - 1]
      const nextGeneration = Generation.next()
      if (generations.length >= 3) { generations.shift() }
      generations.push(nextGeneration)
      return nextGeneration
    }
  }

  namespace Generation {

    /** Returns the next generation. */
    export function next(): NeuralNetwork[] {
      const population = rank(generations[generations.length - 1])
      const nextGen = []

      // Push some of the elite chromosomes into the next generation unchanged.
      for (let i = 0; i < Math.round(elitismRate * populationCount); i++) {
        if (nextGen.length < populationCount) { nextGen.push(population[i]) }
      }

      // Introduce some newly generated chromosomes into the next generation.
      for (let i = 0; i < Math.round(randomnessRate * populationCount); i++) {
        if (nextGen.length < populationCount) { nextGen.push(new NeuralNetwork([2, [2], 1])) }
      }

      // Breed new children from the elites until we hit the population limit.
      for (let ceiling = 1; ceiling < populationCount; ceiling++) {
        for (let i = 0; i < ceiling; i++) {
          const children = breed(population[i], population[ceiling])
          for (const child of children) { nextGen.push(child) }
          if (nextGen.length >= populationCount) { break }
        }
        if (nextGen.length >= populationCount) { break }
      }

      return nextGen
    }

    /** Sorts the population in descending order, ranked by score. */
    function rank(population: NeuralNetwork[]): NeuralNetwork[] {
      const mapped = population.map((chromosome, index) => {
        return { index, score: chromosome.fitness }
      })

      mapped.sort((a, b) => {
        if (a.score > b.score) { return -1 }
        if (a.score < b.score) { return 1 }
        return 0
      })

      return mapped.map((chromosome) => { return population[chromosome.index] })
    }

    /** Takes a pair of parents, performs crossover and mutation and returns the offspring. */
    function breed(parentOne: NeuralNetwork, parentTwo: NeuralNetwork, numberOfOffspring = 1): NeuralNetwork[] {
      // tslint:disable-next-line:no-any
      function crossover(parentOne: any, parentTwo: any): any {
        const child = parentOne

        for (const i in parentTwo.weights) {
          if (Math.random() <= crossoverRange) { child.weights[i] = parentTwo.weights[i] }
        }
        return child
      }

      function mutate(data: IData) {
        for (const i in data.weights) {
          if (Math.random() <= mutationRate) {
            data.weights[i] += Math.random() * mutationRange * 2 - mutationRange
          }
        }
      }

      const offspring = new Array<NeuralNetwork>(numberOfOffspring)

      for (let i = 0; i < numberOfOffspring; i++) {
        const childData = crossover(parentOne.weights, parentTwo.weights)
        mutate(childData)
        const child = new NeuralNetwork()
        child.persist(childData)
        offspring[i] = child
      }

      return offspring
    }
  }
}