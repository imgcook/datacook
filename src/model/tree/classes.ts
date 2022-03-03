import { Tree } from "./tree";
import { BaseEstimator } from "../base";
import { argMax, RecursiveArray, tensor, Tensor, Tensor1D, Tensor2D } from "@tensorflow/tfjs-core";
import { checkArray, checkJSArray } from "../../utils/validation";
import { LabelEncoder } from "../../preprocess/encoder";

export type DecisionTreeCriterion = 'entropy' | 'gini';
export type DecisionTreeSplitter = 'best' | 'random';

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
  public maxFeatures: number;
  public maxLeafNodes: number;
  public minImpurityDecrease: number;
  public tree: Tree;
  public ccpAlpha: number;
  public labelEncoder: LabelEncoder;
  constructor(params: BaseDecisionTreeParams) {
    const { criterion = 'entropy', spliiter = 'best', maxDepth, minSamplesSplit } = params;
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
    const xTensor = checkArray(xData, 'float32', 2);
    const isClassification = this.isClassifier();
    if (isClassification) {
      const yTensor = checkArray(yData, 'any', 1);
      this.labelEncoder = new LabelEncoder();
      this.labelEncoder.init(yTensor);
      const yEncoded = await (await this.labelEncoder.encode(yTensor)).array();
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
