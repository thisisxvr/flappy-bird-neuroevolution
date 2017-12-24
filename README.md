# Flappy Bird Neuroevolution
Neural networks are evolved to learn how to play *Flappy Bird*.

[Demo](https://thisisxvr.github.io/flappy-bird-neuroevolution/).

## Implementation
Over successive generations, the GA evolves neural networks to learn to play the game. 

### Genetic Algorithm
The algorithm initializes with a population of 50 by default. Each bird is controlled by a dedicated neural net. At the end of a generation, a few of the best performers are pushed into the next generation unchanged (determined by the `elitismRate`), along with a few newly generated nets (`randomnessRate`), and new nets (offspring) formed by breeding the elites. 

#### Operators
- Selection: Elitist
- Crossover: Uniform
- Mutation: Uniform

### Neural Network
Each network takes 2 inputs: the bird's position on the y-axis and the position of the lower pipe on the y-axis. It then outputs a value between 0 and 1. If the value is greater than 0.5, the bird's wings are flapped. 

#### Topology
We use a simple fully connected feedforward neural network with 1 input, hidden, and output layer each. The input and hidden layer have 2 neurons while the output layer consists of 1 neuron. The activation function is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function). 

## Literature
- [Neuroevolution](https://www.scholarpedia.org/article/Neuroevolution)
- [Genetic Algorithms](http://www.scholarpedia.org/article/Genetic_algorithms)
- [Feedforward Neural Network](https://en.wikipedia.org/wiki/Feedforward_neural_network)

## Installation
1. `yarn install`
2. `yarn build`
3. `open index.html`

## Credits
This project is a re-implementation of xviniette's [FlappyLearning](https://github.com/xviniette/FlappyLearning). Game assets and code were recycled while the algorithm and neural net were implemented from scratch.

## License
MIT