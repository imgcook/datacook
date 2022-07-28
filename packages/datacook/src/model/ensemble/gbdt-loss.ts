import { Tree } from "../tree";

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
  abstract updateTerminalRegions(
    tree: Tree,
    xData: number[][],
    yData: number[][],
    residual: number[],
    predictions: number[][],
    sampleWeight?: number[],
    sampleMask?: number[],
    learningRate?: number,
    k?: number
  ): void;

  /**
   * Update terminal region for a given leaf
   */
  abstract updateTerminalRegion(
    tree: Tree,
    terminalRegions: number[][],
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
