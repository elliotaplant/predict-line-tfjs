// Use tensorflow.js to do SGD to find the best fit line for a set of points
class Brain {
  constructor() {
    this.width = 600;
    this.height = 400;
    this.points = [];
    this.learningRate = 0.02;
    this.learn = this.learn.bind(this);
  }

  init() {
    this.m = tf.variable(tf.scalar(random(-1, 1)));
    this.b = tf.variable(tf.scalar(random(0.2, 0.8)));
  }

  createCanvas() {
    createCanvas(this.width, this.height);
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
    tf.tidy(() => {
      if (this.points.length <= 0) {
        return;
      }
      if (Number(learningRate.value) !== this.learningRate)  {
        this.learningRate = Number(learningRate.value);
        this.optimizer = tf.train.sgd(this.learningRate);
      }
      this.optimizer.minimize(() => this.loss());
    });
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
}
function keyPressed() {
  if (keyCode === 32) {
    brain.learn();
  }
}

function draw() {
  tf.tidy(() => {
    if (autoFit.checked) {
      brain.learn();
    }
    background(0);
    stroke(255);
    brain.drawPoints();
    brain.drawLine();
  });
}

function toggleFitBtn() {
  if (autoFit.checked) {
    learnBtn.style.display = 'none';
  } else {
    learnBtn.style.display = 'block';
  }
}

const brain = new Brain();

function setup() {
  brain.init();
  brain.createCanvas();
  learnBtn.onclick = brain.learn;
  autoFit.onchange = toggleFitBtn;
}
