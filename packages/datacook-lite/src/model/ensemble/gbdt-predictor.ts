import { OneHotEncoder } from "../../preprocess/encoder";
import { checkJsArray2D } from "../../utils/validation";
import { BaseEstimator } from "../base";
import { Tree } from "../tree/tree";
import { LossFunction, ClassificationLossFunction, LeastSquaredError, MultinomialDeviance, BinomialDeviance } from "./gbdt-loss";
import { predictStages } from "./utils";
import {
  DecisionTreeCriterion,
  DecisionTreeRegressorPredictor
} from "../tree/decision-tree-predictor";
// import { getInverseMask, predictStages, randomSampleMask, trainTestSplit } from "./utils";

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

export class GradientBoostingDecisionTreePredictor extends BaseEstimator {
  protected loss: GBLoss;
  protected learningRate: number;
  protected nEstimators: number;
  protected criterion: DecisionTreeCriterion;
  protected minSamplesSplit: number;
  protected minSamplesLeaf: number;
  protected minWeightFractionLeaf: number;
  protected maxDepth: number;
  protected minImpurityDecrease: number;
  protected init: number;
  protected subSample: number;
  protected maxFeatures: number;
  protected ccpAlpha: number;
  protected alpha: number;
  protected verbose: 1 | 0;
  protected maxLeafNodes: number;
  protected validationFraction: number;
  protected tol: number;
  protected lossFunction: LossFunction;
  protected estimators: DecisionTreeRegressorPredictor[][];
  protected oneHotEncoder: OneHotEncoder<number | string>;
  public trainScore: number[];
  public oobImprovement: number[];


  // Used to decide if early stopping will be used to terminate training
  protected nIterNoChange: number;
  protected lossHistory: number[];
  protected nClass: number;


  constructor(params: GradientBoostingDecisionTreeParams) {
    super();
    const {
      learningRate = 0.1,
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
      nEstimators = 20,
      init,
      subSample,
      verbose
    } = params;

    this.learningRate = learningRate ? learningRate : 0.1;
    if (loss) this.loss = loss;
    if (criterion) this.criterion = criterion;
    if (minSamplesLeaf) this.minSamplesLeaf = minSamplesLeaf;
    if (minImpurityDecrease) this.minImpurityDecrease = minImpurityDecrease;
    if (minWeightFractionLeaf) this.minWeightFractionLeaf = minWeightFractionLeaf;
    if (minSamplesSplit) this.minSamplesSplit = minSamplesSplit;
    if (alpha) this.alpha = alpha;
    if (ccpAlpha) this.ccpAlpha = ccpAlpha;
    if (maxDepth) this.maxDepth = maxDepth;
    if (maxFeatures) this.maxFeatures = maxFeatures;
    if (maxLeafNodes) this.maxLeafNodes = maxLeafNodes;
    if (tol) this.tol = tol;
    if (nIterNoChange) this.nIterNoChange = nIterNoChange;
    if (nEstimators) this.nEstimators = nEstimators;
    if (init) this.init = init;
    if (subSample) this.subSample = subSample;
    if (verbose) this.verbose = verbose;
  }

  private initState() {
    this.estimators = [];
    this.trainScore = [];
    if (this.nIterNoChange)
      this.lossHistory = new Array(this.nIterNoChange).fill(Number.MAX_SAFE_INTEGER);
  }

  /**
   * Get predictions for current estimators
   */
  private stagedPredict(x: number[][]): number[][] {
    const predictions = this.predictionInit(x);
    for (let i = 0; i < this.estimators.length; i++) {
      predictStages(x, this.estimators[i], this.learningRate, predictions);
    }
    return predictions;
  }
  public async predict(x: number[][]): Promise<(number | string | boolean)[]> {
    const predictions = this.stagedPredict(x);
    if (this.isClassifier()) {
      const idx = (this.lossFunction as ClassificationLossFunction).predictionToDecision(predictions);
      const categories = this.oneHotEncoder.categories;
      return idx.map((d) => categories[d]);
    } else {
      return predictions.map((d) => d[0]);
    }
  }

  private predictionInit(x: number[][]): number[][] {
    /**
     * TODO: different init
     */
    const predictions: number[][] = [];
    for (let i = 0; i < x.length; i++) {
      predictions.push(new Array(this.nClass ? this.nClass > 2 ? this.nClass : 1 : 1).fill(0));
    }
    // this.lossFunction.getInitPredictions(x)
    return predictions;
  }

  /**
   * Load model from json string
   * @param modelJson string of model json
   */
  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    const {
      learningRate,
      loss,
      criterion,
      minSamplesLeaf,
      nEstimators,
      minSamplesSplit,
      minWeightFractionLeaf,
      maxDepth,
      minImpurityDecrease,
      init,
      subSample,
      maxFeatures,
      ccpAlpha,
      alpha,
      verbose,
      maxLeafNodes,
      validationFraction,
      tol,
      nIterNoChange,
      trees,
      nClass,
      nFeature,
      classes,
      estimatorType } = modelParams;
    if (learningRate) this.learningRate = learningRate;
    if (loss) this.loss = loss;
    if (criterion) this.criterion = criterion;
    if (minSamplesLeaf) this.minSamplesLeaf = minSamplesLeaf;
    if (nEstimators) this.nEstimators = nEstimators;
    if (minSamplesSplit) this.minSamplesSplit = minSamplesSplit;
    if (minWeightFractionLeaf) this.minWeightFractionLeaf = minWeightFractionLeaf;
    if (maxDepth) this.maxDepth = maxDepth;
    if (minImpurityDecrease) this.minImpurityDecrease = minImpurityDecrease;
    if (init) this.init = init;
    if (subSample) this.subSample = subSample;
    if (maxFeatures) this.maxFeatures = maxFeatures;
    if (ccpAlpha) this.ccpAlpha = ccpAlpha;
    if (alpha) this.alpha = alpha;
    if (verbose) this.verbose = verbose;
    if (maxLeafNodes) this.maxLeafNodes = maxLeafNodes;
    if (validationFraction) this.validationFraction = validationFraction;
    if (tol) this.tol = tol;
    if (nClass) this.nClass = nClass;
    if (nFeature) this.nFeature = nFeature;
    if (nIterNoChange) this.nIterNoChange = nIterNoChange;
    if (estimatorType) this.estimatorType = estimatorType;
    if (classes) {
      this.oneHotEncoder = new OneHotEncoder();
      await this.oneHotEncoder.init(classes);
    }
    if (this.loss == 'ls') {
      this.lossFunction = new LeastSquaredError();
    }
    if (this.loss == 'deviance') {
      if (this.nClass > 2) {
        this.lossFunction = new MultinomialDeviance(this.nClass);
      } else {
        this.lossFunction = new BinomialDeviance();
      }
    }
    if (trees) {
      this.estimators = [];
      for (let i = 0; i < trees.length; i++) {
        const stagedEstimators: DecisionTreeRegressorPredictor[] = [];
        for (let j = 0; j < trees[i].length; j++) {
          const tree = new DecisionTreeRegressorPredictor({
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
          tree.tree = new Tree();
          await tree.tree.fromJson(trees[i][j]);
          stagedEstimators.push(tree);
        }
        this.estimators.push(stagedEstimators);
      }
    }
  }

  public async toJson(): Promise<string> {
    const trees: string[][] = [];
    for (let i = 0; i < this.estimators.length; i++) {
      const stageTrees: string[] = [];
      for (let j = 0; j < this.estimators[i].length; j++) {
        stageTrees.push(await this.estimators[i][j].tree.toJson());
      }
      trees.push(stageTrees);
    }
    const modelParams = {
      learningRate: this.learningRate,
      loss: this.loss,
      criterion: this.criterion,
      minSamplesLeaf: this.minSamplesLeaf,
      nEstimators: this.nEstimators,
      minSamplesSplit: this.minSamplesSplit,
      minWeightFractionLeaf: this.minWeightFractionLeaf,
      maxDepth: this.maxDepth,
      minImpurityDecrease: this.minImpurityDecrease,
      init: this.init,
      subSample: this.subSample,
      maxFeatures: this.maxFeatures,
      ccpAlpha: this.ccpAlpha,
      alpha: this.alpha,
      verbose: this.verbose,
      maxLeafNodes: this.maxLeafNodes,
      validationFraction: this.validationFraction,
      tol: this.tol,
      nIterNoChange: this.nIterNoChange,
      estimatorType: this.estimatorType,
      nClass: this.nClass,
      nFeature: this.nFeature,
      k: this.lossFunction.k,
      trees,
      classes: this?.oneHotEncoder?.categories
    };
    return JSON.stringify(modelParams);
  }
}

/**
 * Gradient boosting model for classification task
 */
export class GradientBoostingClassifierPredictor extends GradientBoostingDecisionTreePredictor {
  /**
   * @param params model parameters
   * Option in params
   * -------
   * `learningRate`: number, default = 0.1
   *     Learning rate
   *
   * `nEstimators`: number, default = 100
   *     Number of ensemble estimators
   *
   * `subSample`: number, default = 1
   *     The fraction of samples to be used for fitting the individual base
   *     learners
   *
   * `tol` : number, default=1e-4
   *     Tolerance for the early stopping. When the loss is not improving
   *     by at least tol for ``nIterNoChange`` iterations (if set to a
   *     number), the training stops.
   *
   * `validation_fraction` : float, default=0.1
   *     The proportion of training data to set aside as validation set for
   *     early stopping. Must be between 0 and 1.
   *     Only used if `nIterNoChange` is set to an integer.
   *
   *
   *  `nIterNoChange` : number, default=none
   *     Used to decide if early stopping will be used
   *     to terminate training when validation score is not improving. By
   *     default it is set to `none` to disable early stopping. If set to a
   *     number, it will set aside `validation_fraction` size of the training
   *     data as validation and terminate training when validation score is not
   *     improving in all of the previous `nIterNoChange` numbers of
   *     iterations. The split is stratified.
   *
   * `minSamplesSplit`: number, default = 2
   *     Minimum number of samples required to split an internal node
   *
   * `minSamplesLeaf`: number, default = 1
   *     The minimum number of samples required to be at leaf node
   *
   * `minWeightFractionLeaf`: number, default = 0
   *     The minimum weighted fraction of the sum total of weights required
   *     to be at a leaf node
   *
   * `maxDepth`: number, default = 3
   *     Maximum depth of the individual regression tree
   *
   * `minImpurityDecrease`: number, default = 0
   *     A node will be split if this split induces a decrease of the impurity
   *     greater than or equal to this value.
   *
   * `maxFeatures` : number, or {"auto", "sqrt", "log2"}, default = None
   *     The number of features to consider when looking for the best split:
   *
   *     - If integer value, then consider `max_features` features at each split.
   *     - If float value, then `max_features` is a fraction and
   *        `int(maxFeatures * n_features)` features are considered at each
   *         split.
   *     - If "auto", then `maxFeatures=sqrt(nFeatures)`.
   *     - If "sqrt", then `maxFeatures=sqrt(nFeatures)`.
   *     - If "log2", then `maxFeatures=log2(nFeatures)`.
   *     - If none, then `maxFeatures=nFatures`.
   *
   * `ccpAlpha`: number, default=0
   *     Complexity parameter used for Minimal Cost-complexity Pruning.
   *     The subtree with the largest cost complexity that is smaller
   *     than `ccpAlpha` will be chosen. By default, no pruning is
   *     performed.
   */
  constructor(params: GradientBoostingDecisionTreeParams) {
    super(params);
    this.estimatorType = 'classifier';
    if (!params.loss) {
      this.loss = 'deviance';
    }
  }
  async predictProba(xData: number[][]): Promise<number[][]> {
    checkJsArray2D(xData);
    this.checkAndSetNFeatures(xData, false);
    return (this.lossFunction as ClassificationLossFunction).predictionToProba(xData);
  }
}

/**
 * Gradient boosting model for regression task
 */
export class GradientBoostingRegressorPredictor extends GradientBoostingDecisionTreePredictor {
  /**
   * @param params model parameters
   * Option in params
   * -------
   * `learningRate`: number, default = 0.1
   *     Learning rate
   *
   * `nEstimators`: number, default = 100
   *     Number of ensemble estimators
   *
   * `subSample`: number, default = 1
   *     The fraction of samples to be used for fitting the individual base
   *     learners
   *
   * `tol` : number, default=1e-4
   *     Tolerance for the early stopping. When the loss is not improving
   *     by at least tol for ``n_iter_no_change`` iterations (if set to a
   *     number), the training stops.
   *
   * `validation_fraction` : float, default=0.1
   *     The proportion of training data to set aside as validation set for
   *     early stopping. Must be between 0 and 1.
   *     Only used if `nIterNoChange` is set to an integer.
   *
   *
   *  `nIterNoChange` : number, default=none
   *     Used to decide if early stopping will be used
   *     to terminate training when validation score is not improving. By
   *     default it is set to `none` to disable early stopping. If set to a
   *     number, it will set aside `validation_fraction` size of the training
   *     data as validation and terminate training when validation score is not
   *     improving in all of the previous `nIterNoChange` numbers of
   *     iterations. The split is stratified.
   *
   * `minSamplesSplit`: number, default = 2
   *     Minimum number of samples required to split an internal node
   *
   * `minSamplesLeaf`: number, default = 1
   *     The minimum number of samples required to be at leaf node,
   *
   * `minWeightFractionLeaf`: number, default = 0
   *     The minimum weighted fraction of the sum total of weights required
   *     to be at a leaf node
   *
   * `maxDepth`: number, default = 3
   *     Maximum depth of the individual regression tree
   *
   * `minImpurityDecrease`: number, default = 0
   *     A node will be split if this split induces a decrease of the impurity
   *     greater than or equal to this value.
   *
   * `max_features` : number, or {"auto", "sqrt", "log2"}, default = None
   *     The number of features to consider when looking for the best split:
   *
   *     - If integer value, then consider `max_features` features at each split.
   *     - If float value, then `max_features` is a fraction and
   *        `int(max_features * n_features)` features are considered at each
   *         split.
   *     - If "auto", then `max_features=sqrt(n_features)`.
   *     - If "sqrt", then `max_features=sqrt(n_features)`.
   *     - If "log2", then `max_features=log2(n_features)`.
   *     - If None, then `max_features=n_features`.
   *
   * `ccpAlpha`: number, default=0
   *     Complexity parameter used for Minimal Cost-complexity Pruning.
   *     The subtree with the largest cost complexity that is smaller
   *     than `ccpAlpha` will be chosen. By default, no pruning is
   *     performed.
   */
  constructor(params: GradientBoostingDecisionTreeParams) {
    super(params);
    this.estimatorType = 'regressor';
    if (!params.loss) {
      this.loss = 'ls';
    }
  }
}
