var Network;
(function (Network) {
    class NeuralNetwork {
        constructor([input, ...hidden]) {
            const output = hidden.pop();
            this.layers = [], this._weights = [];
            let nodesInPreviousLayer = 0, index = 0;
            // Input layer.
            this.layers[0] = new Layer(input, nodesInPreviousLayer, index);
            nodesInPreviousLayer = input;
            // Hidden layers.
            for (const i in hidden) {
                const nodeCount = hidden[i];
                this.layers.push(new Layer(nodeCount, nodesInPreviousLayer, ++index));
                nodesInPreviousLayer = hidden[i];
            }
            // Output layer.
            this.layers.push(new Layer(output, nodesInPreviousLayer, ++index));
        }
        // Returns a flat array with weights of ALL neurons in the network.
        get Weights() {
            if (this._weights.length) {
                return this._weights;
            }
            for (const layer of this.layers) {
                for (const neuron of layer) {
                    this._weights.concat(neuron.weights);
                }
            }
            return this._weights;
        }
        // Persists the mutated weights to the neurons.
        persist() {
            // tslint:disable-next-line:prefer-for-of
            let index = 0;
            for (const layer of this.layers) {
                for (const neuron of layer) {
                    for (const w in neuron.weights) {
                        neuron.weights[w] = this._weights[index];
                        index++;
                    }
                }
            }
        }
        // Computes the output of the network.
        compute(inputs) {
            // Pass inputs to the input layer.
            const inputLayer = this.layers[0];
            for (const i in inputs) {
                for (const inputNeuron of inputLayer) {
                    inputNeuron.value = inputs[i];
                }
            }
            // Compute outputs of hidden layers and output layer.
            let previousLayer = inputLayer;
            for (let i = 1; i < this.layers.length; i++) {
                const layer = this.layers[i];
                for (const neuron of layer) {
                    for (const previousNeuron of previousLayer) {
                        let sum = 0;
                        for (const weight of previousNeuron.weights) {
                            sum += previousNeuron.value * weight;
                        }
                        neuron.activate(sum);
                    }
                }
                previousLayer = layer;
            }
            // Finally, get the output value(s).
            const outputs = [];
            const outputLayer = this.layers[this.layers.length - 1];
            for (const outputNeuron of outputLayer) {
                outputs.push(outputNeuron.value);
            }
            return outputs;
        }
    }
    Network.NeuralNetwork = NeuralNetwork;
    class Layer {
        constructor(nodeCount, inputCount, index = 0) {
            this.pointer = 0;
            this.id = index;
            this.nodes = new Array(nodeCount);
            for (let i = 0; i < nodeCount; i++) {
                this.nodes[i] = new Neuron(inputCount);
            }
        }
        // Returns a Neuron when iterating over a Layer object.
        next() {
            if (this.pointer < this.nodes.length) {
                return { done: false, value: this.nodes[this.pointer++] };
            }
            return { done: true, value: undefined };
        }
        // Allows for iterating over a Layer object.
        [Symbol.iterator]() { return this; }
    }
    class Neuron {
        constructor(weightCount) {
            this.value = undefined;
            this.weights = new Array(weightCount);
            for (let i = 0; i < weightCount; i++) {
                this.weights[i] = Math.random() * 2 - 1;
            }
        }
        // Sigmoid activation function.
        activate(sum) {
            const theta = -sum / 1;
            return this.value = 1 / (1 + Math.exp(theta));
        }
    }
})(Network || (Network = {}));
// import NeuralNetwork from "./neural-network"
// export default GeneticAlgorithm
/// <reference path="neural-network.ts" />
var GeneticAlgorithm;
(function (GeneticAlgorithm) {
    var NeuralNetwork = Network.NeuralNetwork;
    const crossoverRate = 0.5;
    const elitismRate = 0.2;
    const mutationRate = 0.1;
    const mutationRange = 0.5;
    const randomnessRate = 0.2;
    const generations = [];
    let populationCount;
    function evolve(networkShape = [2, [2], 1], popCount = 50) {
        populationCount = popCount;
        if (generations.length === 0) {
            const firstGeneration = new Generation(networkShape, populationCount);
            generations.push(firstGeneration);
            return firstGeneration;
        }
        else {
            const currentGeneration = generations[generations.length - 1];
            const nextGeneration = currentGeneration.next();
            generations.push(nextGeneration);
            return nextGeneration;
        }
    }
    GeneticAlgorithm.evolve = evolve;
    class Generation {
        constructor(networkShape = [2, [2], 1], popCount = populationCount, newPopulation) {
            if (newPopulation) {
                this._population = this.rank(newPopulation);
                return this;
            }
            this._population = new Array(popCount);
            for (let i = 0; i < popCount; i++) {
                this._population[i] = new NeuralNetwork(networkShape);
            }
        }
        get Population() { return this._population; }
        // Sorts the population in descending order, ranked by score.
        rank(population = this._population) {
            const mapped = population.map((chromosome, index) => {
                return { index, score: chromosome.fitness };
            });
            mapped.sort((a, b) => {
                if (a.score > b.score) {
                    return 1;
                }
                if (a.score < b.score) {
                    return -1;
                }
                return 0;
            });
            return mapped.map((chromosome) => { return population[chromosome.index]; });
        }
        // Takes a pair of parents, performs crossover and mutation and returns the offspring.
        breed(parentOne, parentTwo, numberOfOffspring = 1) {
            function crossover(parentOne, parentTwo) {
                const child = parentOne;
                for (const i in parentTwo.Weights) {
                    if (Math.random() <= crossoverRate) {
                        child.Weights[i] = parentTwo.Weights[i];
                    }
                }
                return child;
            }
            function mutate(chromosome) {
                for (const i in chromosome.Weights) {
                    if (Math.random() <= mutationRate) {
                        chromosome.Weights[i] += Math.random() * mutationRange * 2 - mutationRange;
                    }
                }
            }
            const offspring = new Array(numberOfOffspring);
            for (let i = 0; i < numberOfOffspring; i++) {
                const child = crossover(parentOne, parentTwo);
                mutate(child);
                child.persist();
                offspring[i] = child;
            }
            return offspring;
        }
        // Returns the next generation.
        next() {
            const nextGen = [];
            this.rank();
            // Push some of the elite chromosomes into the next generation unchanged.
            for (let i = 0; i < Math.round(elitismRate * populationCount); i++) {
                if (nextGen.length < populationCount) {
                    nextGen.push(this._population[i]);
                }
            }
            // Introduce some randomness into the next generation.
            for (let i = 0; i < Math.round(randomnessRate * populationCount); i++) {
                const firstNetwork = this._population[0];
                for (const i in firstNetwork.Weights) {
                    if (nextGen.length < populationCount) {
                        nextGen.push(firstNetwork.Weights[i] = this.randomClamped());
                    }
                }
            }
            // Breed new children from the elites until we hit the population limit.
            // let ceiling = 1
            for (let ceiling = 1; ceiling < populationCount; ceiling++) {
                for (let i = 0; i < ceiling; i++) {
                    const children = this.breed(this._population[i], this._population[ceiling]);
                    for (const child of children) {
                        nextGen.push(child);
                    }
                    if (nextGen.length >= populationCount) {
                        break;
                    }
                }
                if (nextGen.length >= populationCount) {
                    break;
                }
                // ceiling++
                // if (ceiling >= this._population.length - 1) { ceiling = 0 }
            }
            return new Generation(undefined, undefined, nextGen);
        }
        // Returns a random value between -1 and 1.
        randomClamped() { return Math.random() * 2 - 1; }
    }
    GeneticAlgorithm.Generation = Generation;
})(GeneticAlgorithm || (GeneticAlgorithm = {}));
/// <reference path="genetic-algorithm.ts" />
/// <reference path="neural-network.ts" />
var FlappyBird;
(function (FlappyBird) {
    const canvas = document.querySelector('#flappy-bird');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const backgroundSpeed = 0.5;
    let backgroundx = 0;
    const spawnInterval = 90;
    // let generationNumber = 0
    const fps = 60;
    // tslint:disable-next-line:no-any
    let population, images, score, interval, pipes, birds;
    function start() {
        interval = 0;
        score = 0;
        pipes = [];
        birds = [];
        population = GeneticAlgorithm.evolve().Population;
        for (const _ in population) {
            birds.push(new Bird());
        }
        // generationNumber++
    }
    function display() {
        ctx.clearRect(0, 0, width, height);
        // Draw background.
        for (let i = 0; i < Math.ceil(width / images.bg.width) + 1; i++) {
            ctx.drawImage(images.bg, i * images.bg.width - Math.floor(backgroundx % images.bg.width), 0);
        }
        // Draw pipes.
        for (const i in pipes) {
            const pipe = pipes[i];
            if (Number(i) % 2 === 0) {
                ctx.drawImage(images.pipeTop, pipe.x, pipe.y + pipe.height + images.pipeTop.height, pipe.width, images.pipeTop.height);
            }
            else {
                ctx.drawImage(images.pipeBottom, pipe.x, pipe.y, pipe.width, images.pipeBottom.height);
            }
        }
        ctx.fillStyle = "#FFC600";
        ctx.strokeStyle = "#CE9E00";
        // Draw birds.
        for (const i in birds) {
            const bird = birds[i];
            if (bird.alive) {
                ctx.save();
                ctx.translate(bird.x + bird.width / 2, bird.y + bird.height / 2);
                ctx.rotate(Math.PI / 2 * bird.gravity / 20);
                ctx.drawImage(images.bird, -bird.width / 2, -bird.height / 2, bird.width, bird.height);
                ctx.restore();
            }
        }
        requestAnimationFrame(() => { display(); });
    }
    function update() {
        backgroundx += backgroundSpeed;
        let aperturePos = 0;
        // Get the location of the next gap in pipes.
        if (birds.length > 0) {
            for (const pipe of pipes) {
                if (pipe.x + pipe.width > birds[0].x) {
                    aperturePos = pipe.height / height;
                    break;
                }
            }
        }
        // Pass the inputs to each birds network, and act on the output.
        for (const i in birds) {
            const bird = birds[i];
            if (bird.alive) {
                const inputs = [bird.y / height, aperturePos];
                const decision = population[i].compute(inputs)[0];
                if (decision > 0.5) {
                    bird.flap();
                }
                bird.update();
                if (bird.isDead(height, pipes)) {
                    bird.alive = false;
                    population[i].fitness = score;
                    if (gameOver()) {
                        start();
                    }
                }
            }
        }
        // Update pipes.
        for (let i = 0; i < pipes.length; i++) {
            pipes[i].update();
            if (pipes[i].isOutOfViewport()) {
                pipes.splice(Number(i), 1);
                i--;
            }
        }
        if (interval === 0) {
            const birdDelta = 50;
            const apertureSize = 120;
            const aperturePos = Math.round(Math.random() * (height - birdDelta * 2 - apertureSize) + birdDelta);
            pipes.push(new Pipe(width, 0, undefined, aperturePos));
            pipes.push(new Pipe(width, apertureSize + aperturePos, height));
        }
        interval++;
        if (interval === spawnInterval) {
            interval = 0;
        }
        score++;
        setTimeout(() => { update(); }, 1000 / fps);
    }
    function gameOver() {
        for (const bird of birds) {
            if (bird.alive) {
                return false;
            }
        }
        return true;
    }
    function loadImages(sources, callbackfn) {
        let n = 0, loaded = 0;
        const imgs = {};
        for (const i in sources) {
            n++;
            imgs[i] = new Image();
            imgs[i].src = sources[i];
            imgs[i].onload = () => {
                loaded++;
                if (loaded === n) {
                    callbackfn(imgs);
                }
            };
        }
    }
    // function setSpeed(newFPS: string) { fps = parseInt(newFPS, 10) }
    window.onload = () => {
        const sprites = {
            bird: './img/flaby.png',
            bg: './img/bg.png',
            pipeTop: './img/pipetop.png',
            pipeBottom: './img/pipebottom.png'
        };
        function init() {
            start();
            update();
            display();
        }
        loadImages(sprites, (imgs) => {
            images = imgs;
            init();
        });
    };
    class Bird {
        constructor(x = 80, y = 250, height = 30, width = 40, gravity = 0, velocity = 0.3, jump = -6, alive = true) {
            this.x = x;
            this.y = y;
            this.height = height;
            this.width = width;
            this.gravity = gravity;
            this.velocity = velocity;
            this.jump = jump;
            this.alive = alive;
        }
        flap() { this.gravity = this.jump; }
        update() {
            this.gravity += this.velocity;
            this.y += this.gravity;
        }
        isDead(height, pipes) {
            if (this.y >= height || this.y + this.height <= 0) {
                return true;
            }
            for (const pipe of pipes) {
                if (!(this.x > pipe.x + pipe.width ||
                    this.x + this.width < pipe.x ||
                    this.y > pipe.y + pipe.height ||
                    this.y + this.height < pipe.y)) {
                    return true;
                }
            }
            return false;
        }
    }
    class Pipe {
        constructor(x = 0, y = 0, width = 50, height = 40, speed = 3) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.speed = speed;
        }
        update() { this.x -= this.speed; }
        isOutOfViewport() { return this.x + this.width < 0; }
    }
})(FlappyBird || (FlappyBird = {}));
