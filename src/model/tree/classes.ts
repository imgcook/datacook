import { Tree } from "./tree";
import { BaseEstimator } from "../base";
import { argMax, max, RecursiveArray, tensor, Tensor, Tensor1D, Tensor2D } from "@tensorflow/tfjs-core";
import { checkArray, checkJSArray } from "../../utils/validation";
import { LabelEncoder } from "../../preprocess/encoder";
import { BestSplitter, Splitter } from "./splitter";
import { Criterion, EntropyCriterion, GiniCriterion } from "./criterion";
import { DepthFirstTreeBuilder } from "./tree-builder";

export type DecisionTreeCriterion = 'entropy' | 'gini';
export type DecisionTreeSplitter = 'best';

const CRITERIA_CLF = { "gini": GiniCriterion, "entropy": EntropyCriterion };
const DENSE_SPLITTERS = { "best": BestSplitter };

export interface BaseDecisionTreeParams {
  criterion?: DecisionTreeCriterion,
  splitter?: DecisionTreeSplitter,
  maxDepth?: number,
  minSamplesSplit?: number,
  minSamplesLeaf?: number,
  maxFeatures?: number,
  maxLeafNodes?: number,
  minImpurityDecrease?: number,
  ccpAlpha?: number
}

abstract class BaseDecisionTree extends BaseEstimator{
  public criterion: DecisionTreeCriterion;
  public splitter: DecisionTreeSplitter;
  public maxDepth: number;
  public minSamplesSplit: number;
  public minSamplesLeaf: number;
  public minWeightLeaf: number;
  public maxFeatures: number;
  public maxLeafNodes: number;
  public minImpurityDecrease: number;
  public tree: Tree;
  public ccpAlpha: number;
  public labelEncoder: LabelEncoder;
  public nClass: number;
  public nFeature: number;
  constructor(params: BaseDecisionTreeParams) {
    super();
    const { 
      criterion = 'entropy',
      splitter = 'best',
      maxDepth,
      minSamplesSplit 
    } = params;
    this.criterion = criterion;
    this.splitter = splitter;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  public getDepth() {
    return this.tree.maxDepth;
  }
  public getNLeaves() {
    return this.tree.nLeaves;
  }
  public async fit(xData: Tensor | RecursiveArray<number>, yData: Tensor | string[] | number[], sampleWeight = null) {
    if (this.ccpAlpha < 0) {
      throw new RangeError("ccpAlpha must greater than or equal to zero");
    }
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const isClassification = this.isClassifier();
    this.nFeature = xArray[0].length;
    if (isClassification) {
      const yArray = checkJSArray(yData, 'any', 1);
      this.labelEncoder = new LabelEncoder();
      this.labelEncoder.init(yArray);
      const yEncoded = await (await this.labelEncoder.encode(yArray)).array();
      this.nClass = this.labelEncoder.categories.shape[1];
      const criterion = new CRITERIA_CLF[this.criterion];
      const splitter = new BestSplitter(criterion, this.maxFeatures, this.minSamplesLeaf, this.minWeightLeaf);
      const treeBuilder = new DepthFirstTreeBuilder(splitter, this.minSamplesSplit, this.minSamplesLeaf, this.minWeightLeaf, this.maxDepth, this.minImpurityDecrease);
      const tree = new Tree(this.nFeature, this.nClass);
      treeBuilder.build(tree, xArray, yEncoded);
      for (let i = 0; i < tree.nodeCount;i++){
        // if (tree.nodes[i].leftChild == -1 && tree.nodes[i].rightChild 7=== -1) {
        console.log(JSON.stringify(tree.nodes[i]));
        // }
      }
    }
  }
  public async predict(xData: Tensor | number[][]): Promise<Array<number | string>> {
    const isClassification = this.isClassifier();
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    if (isClassification) {
      const res = this.tree.predict(xArray);
      const labelIds = await (argMax(tensor(res)) as Tensor1D ).array();
      const labels = await (await this.labelEncoder.decode(labelIds)).array();
      return labels;
    }
  }
}
