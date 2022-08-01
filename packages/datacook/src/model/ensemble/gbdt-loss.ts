import { DecisionTreeRegressor, Tree } from "../tree";

export interface LossFunctionConstructor {
  new (): LossFunction;
}


export abstract class LossFunction {
  /**
   * k: The number of regression trees to be induced. 1 for regression and binary classification
   * nClasses for multi-class classification
   */
  public k: number;
  public isMultiClass: boolean;
  constructor(k: number) {
    this.k = k;
  }
  /**
   * Copmute the negative gradient
   * @param y target labels
   * @param predictions predictions of the tree at iteration i - 1
   */
  abstract negativeGradient(
    y: number[] | number[][],
    predictions: number[][],
    k?: number,
    sampleWeight?: number[]
  ): number[];
  /**
   * Update the terminal regions
   * @param tree tree object
   * @param xData input data
   * @param yData input labels
   * @param residual residuals
   * @param predictions predictions of the tree at iteration i-1
   * @param sampleWeight weight of each sample
   * @param sampleMask sample mask to be used
   * @param learningRate learning rate
   * @param k index of estimator to be updated
   */
  public updateTerminalRegions(
    tree: Tree,
    xData: number[][],
    yData: number[][],
    residual: number[],
    predictions: number[][],
    sampleWeight?: number[],
    sampleMask?: number[],
    learningRate?: number,
    k?: number
  ): void {
    const terminalRegions = tree.applyNode(xData);
    for (let i = 0; i < tree.nodes.length; i++) {
      if (tree.nodes[i].leftChild != -1) {
        this.updateTerminalRegion(
          tree,
          terminalRegions,
          i,
          xData,
          yData,
          residual,
          predictions,
          sampleWeight,
          sampleMask,
          learningRate,
          k
        );
      }
    }
    for (let i = 0; i < predictions.length; i++) {
      predictions[i][k] += learningRate * tree.nodes[terminalRegions[i]].value[0];
    }
  }

  /**
   * Update terminal region for a given leaf
   */
  abstract updateTerminalRegion(
    tree: Tree,
    terminalRegions: number[],
    leaf: number,
    xData: number[][],
    y: number[][],
    residual: number[],
    predictions: number[][],
    sampleWeight?: number[],
    sampleMask?: number[],
    learningRate?: number,
    k?: number
  ): void;

  abstract getInitPredictions(x: number[][], estimator: any): number[][];
  abstract computeLoss(y: number[][], predictions: number[][], sampleWeight?: number[], sampleMask?: number[]): number;
}


export abstract class RegressionLossFunction extends LossFunction {
  constructor() {
    super(1);
  }
  getInitPredictions(x: number[][], estimator: any): number[][] {
    const predictions = estimator.predict(x) as number[][];
    return predictions;
  }
}

export class LeastSquaredError extends RegressionLossFunction {
  constructor() {
    super();
  }
  negativeGradient(y: number[][], predictions: number[][], k: number, sampleWeight?: number[]): number[] {
    const grad = new Array(y.length).fill(0);
    for (let i = 0; i < y.length; i++) {
      grad[i] = y[i][k] - predictions[i][k];
    }
    return grad;
  }
  updateTerminalRegion(tree: Tree, terminalRegions: number[], leaf: number, xData: number[][], y: number[][], residual: number[], predictions: number[][], sampleWeight?: number[], sampleMask?: number[], learningRate?: number, k?: number): void {
    // for (let i = 0; i < )
    const prediction = tree.predict(xData);
    for (let i = 0; i < y.length; i++) {
      predictions[i][k] += learningRate * prediction[i][0];
    }
    return;
  }
  updateTerminalRegions(tree: Tree, xData: number[][], yData: number[][], residual: number[], predictions: number[][], sampleWeight?: number[], sampleMask?: number[], learningRate?: number, k?: number): void {
    const prediction = tree.predict(xData);
    learningRate = learningRate ? learningRate : 0.1;
    for (let i = 0; i < predictions.length; i++) {
      predictions[i][k] += learningRate * prediction[i][0];
    }
    return;
  }
  computeLoss(y: number[][], predictions: number[][], sampleWeight?: number[], sampleMask?: number[]): number {
    let loss = 0;
    let weightSum = 0;
    for (let i = 0; i < y.length; i++) {
      if (!sampleMask || sampleMask[i]) {
        const weight = sampleWeight ? sampleWeight[i] : 1;
        for (let k = 0; k < predictions[0].length; k++) {
          weightSum += weight;
          loss += weight * Math.pow(y[i][k] - predictions[i][k], 2);
        }
      }
    }
    return loss / weightSum;
  }
}


export abstract class ClassificationLossFunction extends LossFunction {
  abstract predictionToProba(predictions: number[][]): number[][];
  abstract predictionToDecision(predictions: number[][]): number[];
}

export abstract class DevianceLossFunction extends ClassificationLossFunction {
  negativeGradient(y: number[][], predictions: number[][], k = 0, sampleWeight?: number[]): number[] {
    const grads = new Array(y.length);
    for (let i = 0; i < y.length; i++) {
      grads[i] = y[i][k] - 1 / (1 + Math.exp(-predictions[i][k]));
    }
    return grads;
  }
  computeLoss(y: number[][], predictions: number[][], sampleWeight?: number[], sampleMask?: number[]): number {
    let lossSum = 0;
    let weightSum = 0;
    for (let i = 0; i < y.length; i++) {
      for (let j = 0; j < y[0].length; j++) {
        if (!sampleMask || sampleMask[i]) {
          const weight = sampleWeight ? sampleWeight[i] : 1;
          weightSum += weight;
          lossSum += y[i][j] * predictions[i][j] - Math.log(1 + Math.exp(predictions[i][j]));
        }
      }
    }
    return -2 * lossSum / weightSum;
  }
  updateTerminalRegion(tree: Tree, terminalRegions: number[], leaf: number, xData: number[][], y: number[][], residual: number[], predictions: number[][], sampleWeight?: number[], sampleMask?: number[], learningRate?: number, k?: number): void {
    const terminalMask = terminalRegions.map((d) => d == leaf);
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < terminalMask.length; i++) {
      if ((!sampleMask || sampleMask[i]) && terminalMask[i]) {
        const w = sampleWeight ? sampleWeight[i] : 1;
        numerator += w * residual[i];
        denominator += w * (y[i][k] - residual[i]) * (1 - y[i][k] + residual[i]);

      }
    }
    if (Math.abs(denominator) < 1e-150) {
      tree.nodes[leaf].value[0] = 0;
    }
    tree.nodes[leaf].value[0] = numerator / denominator;
  }
}

export class BinomialDeviance extends DevianceLossFunction {
  constructor() {
    super(1);
  }
  getInitPredictions(x: number[][], estimator: any): number[][] {
    const probas = estimator.predictProb(x) as number[][];
    probas.map((ps) => ps.map((p) => Math.log(p / (1 - p))));
    return probas;
  }
  predictionToProba(predictions: number[][]): number[][] {
    const proba = [];
    for (let i = 0; i < predictions.length; i++) {
      const p = 1 / (1 + Math.exp(-predictions[0]));
      proba.push([ 1 - p, p ]);
    }
    return proba;
  }
  predictionToDecision(predictions: number[][]): number[] {
    return predictions.map((d) => d[0] > 0 ? 1 : 0);
  }

}

export class MultinomialDeviance extends DevianceLossFunction {
  constructor(k: number) {
    super(k);
  }

  getInitPredictions(x: number[][], estimator: any): number[][] {
    const probas = estimator.predictProb(x) as number[][];
    probas.map((ps) => ps.map((p) => Math.log(p / (1 - p))));
    return probas;
  }
  predictionToProba(predictions: number[][]): number[][] {
    const proba = [];
    for (let i = 0; i < predictions.length; i++) {
      let e = 0;
      const ps = [];
      for (let j = 0; j < this.k; j++) {
        e += Math.exp(predictions[i][j]);
      }
      for (let j = 0; j < this.k; j++) {
        ps.push(Math.exp(predictions[i][j]) / e);
      }
      proba.push(ps);
    }
    return proba;
  }
  predictionToDecision(predictions: number[][]): number[] {
    return predictions.map((d) => {
      let ind = 0;
      let max = Number.MIN_SAFE_INTEGER;
      for (let i = 0; i < this.k; i++) {
        if (d[i] > max) {
          max = d[i];
          ind = i;
        }
      }
      return ind;
    });
  }
}
