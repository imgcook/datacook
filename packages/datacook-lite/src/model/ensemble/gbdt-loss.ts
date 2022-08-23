import { Tree } from "../tree/tree";
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


export abstract class ClassificationLossFunction extends LossFunction {
    abstract predictionToProba(predictions: number[][]): number[][];
    abstract predictionToDecision(predictions: number[][]): number[];
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

