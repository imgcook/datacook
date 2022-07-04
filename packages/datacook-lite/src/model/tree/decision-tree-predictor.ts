import { Tree } from "./tree";
import { BaseEstimator } from "../base";
import { LabelEncoder } from "../../preprocess/encoder";

export type DecisionTreeCriterion = 'entropy' | 'gini' | 'mse';
export type DecisionTreeSplitter = 'best';
export type MaxFeaturesSelection = 'auto' | 'sqrt' | 'log2';

function argMax(array: number[]) {
  return [].map.call(array, (x: number, i: number) => [ x, i ]).reduce((r: number[], a: number[]) => (a[0] > r[0] ? a : r))[1];
}

export interface BaseDecisionTreeParams {
  criterion?: DecisionTreeCriterion,
  splitter?: DecisionTreeSplitter,
  maxDepth?: number,
  minSamplesSplit?: number,
  minSamplesLeaf?: number,
  maxFeatures?: number | MaxFeaturesSelection,
  maxLeafNodes?: number,
  minImpurityDecrease?: number,
  ccpAlpha?: number,
  minWeightFractionLeaf?: number
}

class BaseDecisionTreePredictor extends BaseEstimator {
  public criterion: DecisionTreeCriterion;
  public splitter: DecisionTreeSplitter;
  public maxDepth: number;
  public minSamplesSplit: number;
  public minSamplesLeaf: number;
  public minWeightLeaf: number;
  public maxFeatures: number | MaxFeaturesSelection;
  public maxLeafNodes: number;
  public minImpurityDecrease: number;
  public tree: Tree;
  public ccpAlpha: number;
  public labelEncoder: LabelEncoder<number | string>;
  public nClass: number;
  public minWeightFractionLeaf: number;

  constructor(params: BaseDecisionTreeParams = {}) {
    super();
    const {
      splitter = 'best',
      maxDepth,
      minSamplesSplit = 2,
      minSamplesLeaf = 1,
      minWeightFractionLeaf = 0,
      maxFeatures,
      maxLeafNodes,
      minImpurityDecrease = 0,
      ccpAlpha = 0
    } = params;
    this.splitter = splitter;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minWeightFractionLeaf = minWeightFractionLeaf;
    this.maxFeatures = maxFeatures;
    this.maxLeafNodes = maxLeafNodes;
    this.minImpurityDecrease = minImpurityDecrease;
    this.ccpAlpha = ccpAlpha;
  }

  public getDepth() {
    return this.tree.maxDepth;
  }
  public getNLeaves() {
    return this.tree.nLeaves;
  }
  public async predict(xData: number[][]): Promise<number[] | string[]> {
    const isClassification = this.isClassifier();
    const res = this.tree.predict(xData);
    if (isClassification) {
      const labelIds = res.map((d: number[]): number => argMax(d));
      const labels = (await this.labelEncoder.decode(labelIds)) as any;
      return labels;
    } else {
      return res.map((d) => d[0]);
    }
  }
  public async toJson(): Promise<string> {
    const modelParams = {
      criterion: this.criterion,
      splitter: this.splitter,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      minWeightLeaf: this.minWeightLeaf,
      maxFeatures: this.maxFeatures,
      maxLeafNodes: this.maxLeafNodes,
      minImpurityDecrease: this.minImpurityDecrease,
      tree: this.tree.toJson(),
      ccpAlpha: this.ccpAlpha,
      classes: this.labelEncoder?.categories,
      estimatorType: this.estimatorType
    };
    return JSON.stringify(modelParams);
  }

  public async fromJson(modelJson: string): Promise<void> {
    const {
      criterion,
      splitter,
      maxDepth,
      minSamplesSplit,
      minSamplesLeaf,
      minWeightLeaf,
      maxFeatures,
      maxLeafNodes,
      minImpurityDecrease,
      tree,
      ccpAlpha,
      classes,
      nClass,
      minWeightFractionLeaf,
      estimatorType
    } = JSON.parse(modelJson);
    this.criterion = criterion ? criterion : this.criterion;
    this.splitter = splitter ? splitter : this.splitter;
    this.maxDepth = maxDepth ? maxDepth : this.maxDepth;
    this.minSamplesSplit = minSamplesSplit ? minSamplesSplit : this.minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf ? minSamplesLeaf : this.minSamplesLeaf;
    this.minWeightLeaf = minWeightLeaf ? minWeightLeaf : this.minWeightLeaf;
    this.maxFeatures = maxFeatures ? maxFeatures : this.maxFeatures;
    this.maxLeafNodes = maxLeafNodes ? maxLeafNodes : this.maxLeafNodes;
    this.minImpurityDecrease = minImpurityDecrease ? minImpurityDecrease : this.minImpurityDecrease;
    this.ccpAlpha = ccpAlpha ? ccpAlpha : this.ccpAlpha;
    this.nClass = nClass ? nClass : this.nClass;
    this.minWeightFractionLeaf = minWeightFractionLeaf ? minWeightFractionLeaf : this.minWeightFractionLeaf;
    this.estimatorType = estimatorType ? estimatorType : this.estimatorType;
    if (tree) {
      this.tree = this.tree ? this.tree : new Tree();
      this.tree.fromJson(tree);
    }
    if (classes && estimatorType === 'classifier') {
      this.labelEncoder = new LabelEncoder();
      await this.labelEncoder.init(classes);
    }
  }
  // public async toJSON(): Promise<string> {
  //   const modelParams = {
  //     criterion: this.criterion,
  //     splitter: this.splitter,
  //     maxDepth: this.maxDepth,
  //     minSamplesSplit: this.minSamplesSplit,
  //     minSamplesLeaf: this.minSamplesLeaf,
  //     minWeightFractionLeaf: this.minWeightFractionLeaf,
  //     maxFeatures: this.maxFeatures,
  //     minImpurityDecrease: this.minImpurityDecrease,
  //     ccpAlpha: this.ccpAlpha
  //   };
  //   return JSON.stringify(modelParams);
  // }

  // public async fromJSON(modelJson: string): Promise<void> {
  //   const modelParams = JSON.parse(modelJson);
  //   const { criterion, spliiter, maxDepth } = modelParams;
  // }
}

export class DecisionTreeClassifierPredictor extends BaseDecisionTreePredictor {
  public estimatorType = 'classifier';
  /**
   *
   * @param params parameters
   * Option in params
   * ------
   * `criterion`: {"gini", "entropy"}, default="gini"
   *     The function to measure the quality of split
   *
   * `maxDepth`: number,
   *     The maximum depth of the tree. If None, the nodes will
   *     expanded until all leaves are pure or until all leaves
   *     contain less than `minSamplesSplit` samples.
   *
   * `minSampleSplit`: number, default=2
   *     The minimum number of samples required to split an internal
   *     node:
   *     - If integer value, then consider `minSamplesSplit` as the minimum
   *       number.
   *     - If float value, then `minSamplesSplit` is a fraction and
   *       `Math.ceil(minSamplesSplit * nSamples)` are the minimum
   *       number of samples for each split.
   *
   * `minSamplesLeaf`: number, default=1
   *     The minimum number of samples required to be at a leaf node.
   *     A split point at any depth will only be considered if it leaves at
   *     least ``min_samples_leaf`` training samples in each of the left and
   *     right branches.  This may have the effect of smoothing the model,
   *     especially in regression.
   *
   *     - If integer value, then consider `min_samples_leaf` as the minimum number.
   *     - If float value, then `min_samples_leaf` is a fraction and
   *       `ceil(min_samples_leaf * n_samples)` are the minimum
   *       number of samples for each node.
   *
   * `min_weight_fraction_leaf` : number, default=0.0
   *     The minimum weighted fraction of the sum total of weights (of all
   *     the input samples) required to be at a leaf node. Samples have
   *     equal weight when sample_weight is not provided.
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
  constructor(params: BaseDecisionTreeParams = {}) {
    super(params);
    const { criterion = 'gini' } = params;
    this.criterion = criterion;
  }

  public async predictProb(X: number[][]): Promise<number[][]> {
    // const xArray = checkJSArray(X, 'float32', 2) as number[][];
    return this.tree.predict(X);
  }
}

export class DecisionTreeRegressorPredictor extends BaseDecisionTreePredictor {
  public estimatorType = 'regressor';
  constructor(params: BaseDecisionTreeParams = {}) {
    super(params);
    const { criterion = 'mse' } = params;
    this.criterion = criterion;
  }
}
