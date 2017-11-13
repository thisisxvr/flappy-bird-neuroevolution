/// <reference path="genetic-algorithm.ts" />
/// <reference path="neural-network.ts" />

namespace FlappyBird {
  import NeuralNetwork  = Network.NeuralNetwork
  const canvas          = document.querySelector('#flappy-bird') as HTMLCanvasElement
  const ctx             = canvas.getContext('2d')!
  const width           = canvas.width
  const height          = canvas.height
  const backgroundSpeed = 0.5
  let backgroundx       = 0
  const spawnInterval   = 90
  // let generationNumber = 0
  const fps = 60
  // tslint:disable-next-line:no-any
  let population: NeuralNetwork[], images: {[index: string]: any}, score: number, interval: number, pipes: Pipe[], birds: Bird[]

  function start() {
    interval = 0
    score = 0
    pipes = []
    birds = []

    population = GeneticAlgorithm.evolve().Population
    for (const _ in population) { birds.push(new Bird()) }
    // generationNumber++
  }

  function display() {
    ctx.clearRect(0, 0, width, height)

    // Draw background.
    for (let i = 0; i < Math.ceil(width / images.bg.width) + 1; i++) {
      ctx.drawImage(images.bg, i * images.bg.width - Math.floor(backgroundx % images.bg.width), 0)
    }

    // Draw pipes.
    for (const i in pipes) {
      const pipe = pipes[i]
      if (Number(i) % 2 === 0) {
        ctx.drawImage(images.pipeTop,
          pipe.x, pipe.y + pipe.height + images.pipeTop.height,
          pipe.width, images.pipeTop.height) }
      else {
        ctx.drawImage(images.pipeBottom,
          pipe.x, pipe.y, pipe.width, images.pipeBottom.height)
      }
    }

    ctx.fillStyle = "#FFC600"
    ctx.strokeStyle = "#CE9E00"

    // Draw birds.
    for (const i in birds) {
      const bird = birds[i]
      if (bird.alive) {
        ctx.save()
        ctx.translate(bird.x + bird.width / 2, bird.y + bird.height / 2)
        ctx.rotate(Math.PI / 2 * bird.gravity / 20)
        ctx.drawImage(images.bird, -bird.width / 2, -bird.height / 2, bird.width, bird.height)
        ctx.restore()
      }
    }

    requestAnimationFrame(() => { display() })
  }

  function update() {
    backgroundx += backgroundSpeed
    let aperturePos = 0

    // Get the location of the next gap in pipes.
    if (birds.length > 0) {
      for (const pipe of pipes) {
        if (pipe.x + pipe.width > birds[0].x) {
          aperturePos = pipe.height / height
          break
        }
      }
    }

    // Pass the inputs to each birds network, and act on the output.
    for (const i in birds) {
      const bird = birds[i]
      if (bird.alive) {
        const inputs = [bird.y / height, aperturePos]
        const decision = population[i].compute(inputs)[0]
        if (decision > 0.5) { bird.flap() }

        bird.update()
        if (bird.isDead(height, pipes)) {
          bird.alive = false
          population[i].fitness = score
          if (gameOver()) { start() }
        }
      }
    }

    // Update pipes.
    for (let i = 0; i < pipes.length; i++) {
      pipes[i].update()
      if (pipes[i].isOutOfViewport()) {
        pipes.splice(Number(i), 1)
        i--
      }
    }

    if (interval === 0) {
      const birdDelta = 50
      const apertureSize = 120
      const aperturePos = Math.round(Math.random() * (height - birdDelta * 2 - apertureSize) + birdDelta)

      pipes.push(new Pipe(width, 0, undefined, aperturePos))
      pipes.push(new Pipe(width, apertureSize + aperturePos, height))
    }

    interval++
    if (interval === spawnInterval) { interval = 0 }

    score++
    setTimeout(() => { update() }, 1000 / fps)
  }

  function gameOver(): boolean {
    for (const bird of birds) {
      if (bird.alive) { return false }
    }
    return true
  }

  function loadImages(sources: {[index: string]: string}, callbackfn: Function) {
    let n = 0, loaded = 0
    const imgs: { [index: string]: HTMLImageElement } = { }

    for (const i in sources) {
      n++
      imgs[i] = new Image()
      imgs[i].src = sources[i]
      imgs[i].onload = () => {
        loaded++
        if (loaded === n) { callbackfn(imgs) }
      }
    }
  }

  // function setSpeed(newFPS: string) { fps = parseInt(newFPS, 10) }

  window.onload = () => {
    const sprites = {
      bird: './img/flaby.png',
      bg: './img/bg.png',
      pipeTop: './img/pipetop.png',
      pipeBottom: './img/pipebottom.png'
    }

    function init() {
      start()
      update()
      display()
    }

    loadImages(sprites, (imgs: { [index: string]: HTMLImageElement }) => {
      images = imgs
      init()
    })
  }

  class Bird {
    constructor(
      readonly x = 80,
      public y = 250,
      readonly height = 30,
      readonly width = 40,
      public gravity = 0,
      private velocity = 0.3,
      private jump = -6,
      public alive = true
    ) { }

    flap() { this.gravity = this.jump }

    update() {
      this.gravity += this.velocity
      this.y += this.gravity
    }

    isDead(height: number, pipes: Pipe[]): boolean {
      if (this.y >= height || this.y + this.height <= 0) { return true }

      for (const pipe of pipes) {
        if (!(
          this.x > pipe.x + pipe.width  ||
          this.x + this.width < pipe.x  ||
          this.y > pipe.y + pipe.height ||
          this.y + this.height < pipe.y
        )) { return true }
      }

      return false
    }
  }

  class Pipe {
    constructor(
      public x = 0,
      readonly y = 0,
      readonly width = 50,
      readonly height = 40,
      readonly speed = 3
    ) { }

    update() { this.x -= this.speed }

    isOutOfViewport(): boolean { return this.x + this.width < 0 }
  }
}