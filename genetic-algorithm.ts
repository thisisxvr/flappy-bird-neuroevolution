/// <reference path="neural-network.ts" />

namespace GeneticAlgorithm {
  import NeuralNetwork = Network.NeuralNetwork
  const crossoverRate = 0.5
  const elitismRate = 0.2
  const mutationRate = 0.1
  const mutationRange = 0.5
  const randomnessRate = 0.2
  const generations: NeuralNetwork[][] = []
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

    // constructor(networkShape = [2, [2], 1], popCount = 50, newPopulation?: NeuralNetwork[]) {
    //   if (newPopulation) {
    //     population = newPopulation
    //     return this
    //   }

    //   population = new Array<NeuralNetwork>(popCount)
    //   for (let i = 0; i < popCount; i++) {
    //     population[i] = new NeuralNetwork(networkShape)
    //   }
    // }

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
        // const newChromosome = population[0]
        // for (const i in newChromosome.weights) {
        //   newChromosome.weights[i] = randomClamped()
        // }
        // newChromosome.persist()
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

      // const nns = []
      // for (const i in nextGen) {
      //   const nn = new NeuralNetwork()
      //   nn.setSave(nextGen[i])
      //   nns.push(nn)
      // }

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
        child.persist()
        offspring[i] = child
      }

      return offspring
    }

    /** Returns a random value between -1 and 1. */
    function randomClamped(): number { return Math.random() * 2 - 1 }
  }
}