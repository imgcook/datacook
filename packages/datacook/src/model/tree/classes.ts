import { Tree } from "./tree";
<<<<<<< HEAD
import { BaseClassifier, BaseEstimator, ClassMap } from "../base";
import { RecursiveArray, tensor, Tensor, Tensor2D } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../../utils/validation";
import { LabelEncoder, OneHotDropTypes, OneHotEncoder } from "../../preprocess/encoder";
=======
import { BaseEstimator } from "../base";
import { RecursiveArray, Tensor, Tensor2D } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../../utils/validation";
import { LabelEncoder } from "../../preprocess/encoder";
>>>>>>> 29627e940ac3f9106bc2937efc579191b08fc2d9
import { BestSplitter } from "./splitter";
import { EntropyCriterion, GiniCriterion, MSECriterion } from "./criterion";
import { DepthFirstTreeBuilder } from "./tree-builder";
import { buildPrunedTree } from "./tree-pruner";
<<<<<<< HEAD
import { applyMixins } from "../../utils/mix";
=======
>>>>>>> 29627e940ac3f9106bc2937efc579191b08fc2d9

export type DecisionTreeCriterion = 'entropy' | 'gini' | 'mse';
export type DecisionTreeSplitter = 'best';
export type MaxFeaturesSelection = 'auto' | 'sqrt' | 'log2';

const CRITERIA_CLF = { "gini": GiniCriterion, "entropy": EntropyCriterion, 'mse': MSECriterion };
<<<<<<< HEAD
const DENSE_SPLITTERS = { "best": BestSplitter };
=======
// const DENSE_SPLITTERS = { "best": BestSplitter };
>>>>>>> 29627e940ac3f9106bc2937efc579191b08fc2d9


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

class BaseDecisionTree extends BaseEstimator {
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
  public labelEncoder: LabelEncoder;
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
  private getMaxFeatures(): number {
    const isClassification = this.isClassifier();
    if (!this.maxFeatures) {
      return this.nFeature;
    }
    if (typeof this.maxFeatures === 'number') {
      if (Number.isInteger(this.maxFeatures)) {
        return this.maxFeatures;
      } else {
        return Math.max(this.maxFeatures, 1) * this.nFeature;
      }
    } else {
      switch (this.maxFeatures) {
      case 'auto':
        if (isClassification) {
          return Math.max(1, Math.sqrt(this.nFeature));
        } else {
          return this.nFeature;
        }
      case 'sqrt':
        return Math.max(1, Math.sqrt(this.nFeature));
      case 'log2':
        return Math.max(1, Math.log2(this.nFeature));
      default:
        throw new TypeError('Invalid value for maxFeatures, should be "auto", "sqrt" or "log2"');
      }
    }
  }
  private getMinSamplesSplit(nSamples: number): number {
    if (Number.isInteger(this.minSamplesSplit)) {
      if (!(this.minSamplesSplit >= 2)) {
        throw new TypeError('minSamplesSplit should be an integer greater than 1 or in (0.0, 1.0]');
      }
      return this.minSamplesSplit;
    }
    if (!(this.minSamplesLeaf > 0 || this.minSamplesLeaf <= 1)) {
      throw new TypeError('minSamplesSplit should be an integer greater than 1 or in (0.0, 1.0]');
    }
    return Math.max(2, Math.ceil(this.minSamplesLeaf * nSamples));
  }

  /**
   * Fit model according to X, y.
   * @param xData Tensor like of shape (n_samples, n_features), input feature
   * @param yData Tensor like of shape (n_sample, ), input target label
   * @param sampleWeight Sample weights
   */
  public async fit(xData: Tensor | RecursiveArray<number>, yData: Tensor | string[] | number[], sampleWeight: number[] = null) {
    if (this.ccpAlpha < 0) {
      throw new RangeError("ccpAlpha must greater than or equal to zero");
    }
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const isClassification = this.isClassifier();
    const nSamples = xArray.length;
    this.nFeature = xArray[0].length;

    if (this.maxDepth < 0) {
      throw new TypeError('maxDepth should be greater than 0');
    }
    if (this.maxFeatures && (this.maxFeatures <= 0 || this.maxFeatures > this.nFeature)) {
      throw new TypeError('maxFeatures should be in (0, nFeature]');
    }
    if (this.maxLeafNodes && !Number.isInteger(this.maxLeafNodes)) {
      throw new TypeError('maxLeafNodes should be integral number');
    }
    if (this.maxLeafNodes < 1 ) {
      throw new TypeError('maxLeafNodes should be either null or greater than 1');
    }
    if (Number.isInteger(this.minSamplesLeaf)) {
      if (this.minSamplesLeaf < 1) {
        throw new TypeError('minSamplesLeaf should be at least 1 or in (0, 0.5]');
      }
    } else {
      if (!(this.minSamplesLeaf > 0 && this.minSamplesLeaf <= 0.5)) {
        throw new TypeError('minSamplesLeaf should be at least 1 or in (0, 0.5]');
      }
    }
    if (!(this.minWeightFractionLeaf >= 0 && this.minWeightFractionLeaf <= 0.5)) {
      throw new TypeError('minWeightFractionLeaf must in [0, 0.5]');
    }
    if (!(this.minImpurityDecrease >= 0)) {
      throw new TypeError('minImpurityDecreases should be greater or equal to 0');
    }

    const maxDepth = this.maxDepth ? this.maxDepth : Number.MAX_SAFE_INTEGER;
<<<<<<< HEAD
    const maxLeafNodes = this.maxLeafNodes ? this.maxLeafNodes : -1;
=======
    // const maxLeafNodes = this.maxLeafNodes ? this.maxLeafNodes : -1;
>>>>>>> 29627e940ac3f9106bc2937efc579191b08fc2d9
    const minWeightLeaf = sampleWeight ?
      this.minWeightFractionLeaf * sampleWeight.reduce((a: number, b: number): number => a + b) :
      this.minWeightFractionLeaf * nSamples;
    const maxFeatures = this.getMaxFeatures();
    const minSamplesLeaf = Number.isInteger(this.minSamplesLeaf) ?
      this.minSamplesLeaf :
      Math.ceil(this.minSamplesLeaf * nSamples);
    const minSamplesSplit = this.getMinSamplesSplit(nSamples);
    const minImpurityDecrease = this.minImpurityDecrease;

    let y: number[] = [];
    const yArray = checkJSArray(yData, 'any', 1) as any;
    if (isClassification) {
      this.labelEncoder = new LabelEncoder();
      await this.labelEncoder.init(yArray);
      y = await (await this.labelEncoder.encode(yArray)).array() as number[];
      this.nClass = this.labelEncoder.categories.shape[1];
    } else {
      y = yArray;
    }
    const criterion = new CRITERIA_CLF[this.criterion];
    const splitter = new BestSplitter(criterion, maxFeatures, minSamplesLeaf, minWeightLeaf);
    const treeBuilder = new DepthFirstTreeBuilder(
      splitter,
      minSamplesSplit,
      minSamplesLeaf,
      minWeightLeaf,
      maxDepth,
      minImpurityDecrease);
    const tree = new Tree(this.nFeature);
    treeBuilder.build(tree, xArray, y, sampleWeight);
    this.tree = tree;
    this.pruneTree();
  }
  public async predict(xData: Tensor | number[][]): Promise<number[] | string[]> {
    const isClassification = this.isClassifier();
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const res = this.tree.predict(xArray);
    if (isClassification) {
      const labelIds = res.map((d: number[]): number => argMax(d));
      const labels = await (await this.labelEncoder.decode(labelIds)).array() as any;
      return labels;
    } else {
      return res.map((d) => d[0]);
    }
  }
  private pruneTree(): void {
    if (this.ccpAlpha < 0) {
      throw new TypeError("ccpAlpha must be greater than or equal to 0");
    }
    if (this.ccpAlpha === 0) {
      return;
    }
    this.tree = buildPrunedTree(this.tree, this.ccpAlpha);
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
      classes: this.labelEncoder?.categories?.arraySync(),
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
<<<<<<< HEAD
      this.tree.fromJson(tree);
=======
      await this.tree.fromJson(tree);
>>>>>>> 29627e940ac3f9106bc2937efc579191b08fc2d9
    }
    if (classes && estimatorType === 'classifier') {
      this.labelEncoder = new LabelEncoder();
      await this.labelEncoder.init(classes);
    }
  }
}


export class DecisionTreeClassifier extends BaseDecisionTree {
  public estimatorType = 'classifier';
  public validateData: (x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension: number, yDimension: number) => { x: Tensor, y: Tensor };
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

  public async predictProb(X: number[][] | Tensor2D): Promise<number[][]> {
    const xArray = checkJSArray(X, 'float32', 2) as number[][];
    return this.tree.predict(xArray);
  }
}

export class DecisionTreeRegressor extends BaseDecisionTree {
  public estimatorType = 'regressor';
  constructor(params: BaseDecisionTreeParams = {}) {
    super(params);
    const { criterion = 'mse' } = params;
    this.criterion = criterion;
  }
}
