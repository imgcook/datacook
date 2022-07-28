import { LabelEncoder } from "../../preprocess/encoder";
import { OneHotEncoder } from "../../preprocess/encoder";
import { checkJSArray } from "../../utils/validation";
import { BaseEstimator } from "../base";
import { DecisionTreeCriterion, DecisionTreeRegressor } from "../tree/classes";
import { LossFunction } from "./gbdt-loss";
import { getInverseMask, predictStages, randomSampleMask } from "./utils";

export type GBLoss = 'deviance' | 'ls' | 'lad';

export interface GradientBoostingDecisionTreeParams {
  loss?: GBLoss;
  learningRate?: number;
  nEstimators?: number;
  criterion?: DecisionTreeCriterion;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
  minWeightFractionLeaf?: number;
  maxDepth?: number;
  minImpurityDecrease?: number;
  init?: number;
  subSample?: number;
  maxFeatures?: number;
  ccpAlpha?: number;
  alpha?: number;
  verbose?: 1 | 0;
  maxLeafNodes?: number;
  validationFraction?: number;
  tol?: number;
  nIterNoChange?: number;
}

export class GradientBoostingDecisionTree extends BaseEstimator {
  private loss: GBLoss;
  private learningRate: number;
  private nEstimators: number;
  private criterion: DecisionTreeCriterion;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private minWeightFractionLeaf: number;
  private maxDepth: number;
  private minImpurityDecrease: number;
  private init: number;
  private subSample: number;
  private maxFeatures: number;
  private ccpAlpha: number;
  private alpha: number;
  private verbose: 1 | 0;
  private maxLeafNodes: number;
  private validationFraction: number;
  private tol: number;
  private lossFunction: LossFunction;
  private estimators: DecisionTreeRegressor[][];
  private oneHotEncoder: OneHotEncoder;
  public trainScore: number[];
  public oobImprovement: number[];


  // Used to decide if early stopping will be used to terminate training
  public nIterNoChange: number;
  public lossHistory: number[];
  public nClass: number;


  constructor(params: GradientBoostingDecisionTreeParams) {
    super();
    const {
      learningRate,
      loss,
      criterion,
      minSamplesLeaf,
      minImpurityDecrease,
      minWeightFractionLeaf,
      minSamplesSplit,
      alpha,
      ccpAlpha,
      maxDepth,
      maxFeatures,
      maxLeafNodes,
      tol,
      nIterNoChange,
      nEstimators,
      init,
      subSample,
      verbose
    } = params;

    this.learningRate = learningRate;
    this.loss = loss;
    this.criterion = criterion;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minImpurityDecrease = minImpurityDecrease;
    this.minWeightFractionLeaf = minWeightFractionLeaf;
    this.minSamplesSplit = minSamplesSplit;
    this.alpha = alpha;
    this.ccpAlpha = ccpAlpha;
    this.maxDepth = maxDepth;
    this.maxFeatures = maxFeatures;
    this.maxLeafNodes = maxLeafNodes;
    this.tol = tol;
    this.nIterNoChange = nIterNoChange;
    this.nEstimators = nEstimators;
    this.init = init;
    this.subSample = subSample;
    this.verbose = verbose;
  }

  private async fitStage(
    i: number,
    xData: number[][],
    yData: number[][],
    predictions: number[][],
    sampleWeight?: number[],
    sampleMask?: number[],
  ): Promise<number[][]> {
    const lossFunc = this.lossFunction;
    for (let k = 0; k < lossFunc.k; k++) {
      const residuals = lossFunc.negativeGradient(yData, predictions, k, sampleWeight);
      const tree = new DecisionTreeRegressor({
        criterion: this.criterion,
        splitter: 'best',
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        minWeightFractionLeaf: this.minWeightFractionLeaf,
        minImpurityDecrease: this.minImpurityDecrease,
        maxFeatures: this.maxFeatures,
        maxLeafNodes: this.maxLeafNodes,
        ccpAlpha: this.ccpAlpha
      });
      await tree.fit(xData, residuals, sampleWeight);
      lossFunc.updateTerminalRegions(
        tree.tree,
        xData,
        yData,
        residuals,
        predictions,
        sampleWeight,
        sampleMask,
        this.learningRate,
        k
      );
      this.estimators[i][k] = tree;
    }
    return predictions;
  }


  private async fitStages(
    x: number[][],
    y: number[][],
    predictions: number[][],
    sampleWeight: number[],
    beginAtStage = 0,
    xVal?: number[][],
    yVal?: number[][]
  ): Promise<number> {
    const nSamples = x.length;
    const doOOB = this.subSample < 1;
    let sampleMask: number[];
    let inverseSampleMask: number[];
    const nInbag = Math.max(1, this.subSample * nSamples);
    const lossFunc = this.lossFunction;
    let lossHistory: number[];
    let oldOOBScore;
    let i: number;
    let yValPred: number[][];
    /**
     * TODO: verbose
     */
    if (this.nIterNoChange) {
      lossHistory = new Array(this.nIterNoChange).fill(Number.MAX_SAFE_INTEGER);
      yValPred = this.stagedPredict(xVal);
    }
    for (i = beginAtStage; i < this.nEstimators; i++) {
      if (doOOB) {
        sampleMask = randomSampleMask(nSamples, nInbag);
        inverseSampleMask = getInverseMask(sampleMask);
        oldOOBScore = lossFunc.computeLoss(y, predictions, sampleWeight, inverseSampleMask);
      }
      predictions = await this.fitStage(
        i,
        x,
        y,
        predictions,
        sampleWeight,
        sampleMask
      );
      if (doOOB) {
        this.trainScore[i] = lossFunc.computeLoss(y, predictions, sampleWeight, sampleMask);
        this.oobImprovement[i] = oldOOBScore - lossFunc.computeLoss(y, predictions, sampleWeight, inverseSampleMask);
      } else {
        this.trainScore[i] = lossFunc.computeLoss(y, predictions, sampleWeight);
      }
      if (this.nIterNoChange) {
        const validationLoss = lossFunc.computeLoss(yVal, yValPred, sampleWeight);
        for (let j = 0; j < this.lossHistory.length; j++) {
          if (validationLoss + this.tol < lossHistory[j]) {
            this.lossHistory[ i % this.lossHistory.length ] = validationLoss;
            break;
          }
        }
      }
    }
    return i + 1;
  }

  private stagedPredict(x: number[][]): number[][] {
    const predictions = this.predictionInit(x);
    for (let i = 0; i < this.estimators.length; i++) {
      predictStages(x, this.estimators[i], this.learningRate, predictions);
    }
    return predictions;
  }

  private predictionInit(x: number[][]): number[][] {
    /**
     * TODO: different init
     */
    const predictions: number[][] = [];
    for (let i = 0; i < x.length; i++) {
      predictions.push(new Array(this.lossFunction.k).fill(0));
    }
    return predictions;
  }

  public async fit(
    xData: number[][],
    yData: number[] | string[],
    sampleWeight?: number[]
  ): Promise<void> {
    /**
     * TODO: Data Validation
     */
    let y: number[][];
    const isClassification = this.isClassifier();
    const yArray = checkJSArray(yData, 'any', 1) as any;
    if (isClassification) {
      this.oneHotEncoder = new OneHotEncoder();
      await this.oneHotEncoder.init(yArray);
      y = await (await this.oneHotEncoder.encode(yData)).array() as number[][];
      this.nClass = this.oneHotEncoder.categories.shape[1];
    } else {
      y = yArray as number[][];
    }
    const predictions = this.predictionInit(xData);
    const nStages = this.fitStages(
      xData,
      y,
      predictions,
      sampleWeight,
    );
    // if (nStages != this.estimators.length) {

    // }

  }
}
