// Use tensorflow.js to do SGD to find the best fit line for a set of points
class Brain {
  constructor() {
    this.width = 600;
    this.height = 400;
    this.points = [];
    this.learningRate = 0.02;
    this.momentum = 0.02;
    this.optimizerName = 'momentum';
    this.learn = this.learn.bind(this);
    this.numLearns = 0;
    this.prevCost = 0;
    this.costDelta = 1;
  }

  init() {
    this.learningRate = JSON.parse(localStorage.getItem('learningRate') || '0.02');
    autoFit.checked = JSON.parse(localStorage.getItem('autofit') || 'true');
    this.optimizerName = localStorage.getItem('optimizer') || 'sgd';
    this.m = tf.variable(tf.scalar(random(-1, 1)));
    this.b = tf.variable(tf.scalar(random(0.2, 0.8)));
    setImmediate(() => {
      learningRate.value = this.learningRate;
      optimizerSelect.value = this.optimizerName;
      optimizerSelect.value = this.optimizerName;
      renderFitBtn();
    });
  }

  createCanvas() {
    return createCanvas(this.width, this.height);
  }


  addPoint(x, y) {
    const newPoint = fromCanvas(x, y);
    if (newPoint[0] >= 0 && newPoint[0] <= 1 && newPoint[1] >= 0 && newPoint[1] <= 1) {
      this.points.push(newPoint);
    }
  }

  predict(x) {
    return this.m.mul(tf.tensor1d(x)).add(this.b);
  }

  loss() {
    const predictions = this.predict(this.points.map(p => p[0]));
    const actual = tf.tensor1d(this.points.map(p => p[1]));
    return actual.sub(predictions).square().mean();
  }

  learn() {
    if (this.points.length <= 0) {
      return;
    }

    tf.tidy(() => {
      let needsNewOpt = !this.optimizer;

      if (Number(learningRate.value) !== this.learningRate)  {
        this.learningRate = Number(learningRate.value);
        needsNewOpt = true;
      }

      if (optimizerSelect.value !== this.optimizerName)  {
        this.optimizerName = optimizerSelect.value;
        needsNewOpt = true;
      }

      if (needsNewOpt) {
        if (this.optimizerName === 'momentum') {
          this.optimizer = tf.train.momentum(this.learningRate, 0.1);
        } else {
          this.optimizer = tf.train[this.optimizerName](this.learningRate);
        }
      }
    });

    tf.tidy(() => {
      const cost = this.optimizer.minimize(() => this.loss(), true).dataSync();
      this.costDelta = cost[0] - this.prevCost;
      this.prevCost = cost[0];
    });
  }

  closeEnough() {
    return Math.abs(this.costDelta) < 1e-7;
  }

  drawPoints() {
    strokeWeight(8);
    this.points.map(point => toCanvas(...point)).forEach(pt => point(...pt));
  }

  drawLine() {
    strokeWeight(2);
    const lineX = [0, 1];
    const lineY = this.predict(lineX).dataSync();
    line(...toCanvas(lineX[0], lineY[0]), ...toCanvas(lineX[1], lineY[1]));
  }
}


function fromCanvas(x, y) {
  return [map(x, 0, width, 0, 1), map(y, 0, height, 1, 0)];
}

function toCanvas(x, y) {
  return [map(x, 0, 1, 0, width), map(y, 0, 1, height, 0)];
}

function mousePressed() {
  brain.addPoint(mouseX, mouseY);
  brain.learn();
}

function keyPressed() {
  // Check spacebar
  if (!autoFit.checked && keyCode === 32) {
    brain.learn();
  }
}

function draw() {
  tf.tidy(() => {
    if (autoFit.checked && !brain.closeEnough()) {
      brain.learn();
    }
    background(0);
    stroke(255);
    brain.drawPoints();
    brain.drawLine();
  });
}

function toggleFitBtn() {
  localStorage.setItem('autofit', autoFit.checked);
  renderFitBtn();
}

function renderFitBtn() {
  if (autoFit.checked) {
    learnBtn.style.display = 'none';
  } else {
    learnBtn.style.display = 'block';
  }
}

function updateLearningRate() {
  localStorage.setItem('learningRate', learningRate.value);
  brain.learn();
}

function updateOptimizer() {
  localStorage.setItem('optimizer', optimizerSelect.value);
  brain.learn();
}

let brain = new Brain();

function reset() {
  brain = new Brain();
  brain.init();
}

function setup() {
  brain.init();
  const canvas = brain.createCanvas();
  canvas.parent('canvasHolder');

  learnBtn.onclick = brain.learn;
  resetBtn.onclick = reset;
  autoFit.onchange = toggleFitBtn;
  learningRate.onchange = updateLearningRate;
  optimizerSelect.onchange = updateOptimizer;
}
