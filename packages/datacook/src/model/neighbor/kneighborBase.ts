import { add, any, divNoNan, equal, fill, mul, neg, reshape, sum, Tensor1D, Tensor2D, tidy } from "@tensorflow/tfjs-core";
import { checkJSArray } from "../../utils/validation";
import { BallTree } from "./ballTree";
import { BinaryTreeParams } from "./binaryTree";
import { BruteNeighbor, BruteNeighborParams } from "./bruteNeighbor";
import { KDTree } from "./kdTree";
import { MetricName, MetricParams } from "./metrics";
import { NeighborhoodMethod } from "./neighborhood";

export type WeightFunction = (distances: Tensor2D) => Tensor2D;

export const WEIGHT_FUNCTIONS = {
  uniform: (distances: Tensor2D): Tensor2D => {
    const shape = distances.shape;
    return fill(shape, 1 / shape[1]);
  },
  distance: (distances: Tensor2D): Tensor2D => {
    return tidy(() => {
      const isZero = equal(distances, 0);
      // check if contain 0
      const hasZero = any(isZero, 1);
      const inverseDist = divNoNan(1, distances);
      const invSum = sum(inverseDist, 1);
      const weights = divNoNan(inverseDist, reshape(invSum, [ -1, 1 ]));
      const zeroWeights = divNoNan(isZero, reshape(sum(isZero, 1), [ -1, 1 ]));
      return add(mul(weights, reshape(neg(hasZero), [ -1, 1 ])), zeroWeights) as Tensor2D;
    });
  }
};

// export const NEIGHBOR_METHODS = {
//   ballTree: new BallTree(),
//   kdTree: new KDTree(),
//   brute: new BruteNeighbor()
// };

export type NeighborMethodType = 'ballTree' | 'kdTree' | 'brute';
export type NeighborParams = BinaryTreeParams | BruteNeighborParams;

export class NeighborFactory {
  public static getNeighborMethod(name: NeighborMethodType, params: NeighborParams = {}): NeighborhoodMethod {
    switch (name) {
    case 'ballTree':
      return new BallTree(params as BinaryTreeParams);
    case 'kdTree':
      return new KDTree(params as BinaryTreeParams);
    case 'brute':
      return new BruteNeighbor(params as BruteNeighborParams);
    default:
      return new KDTree(params as BinaryTreeParams);
    }
  }
}

export interface KNeighborParams {
  algorithm?: NeighborMethodType;
  leafSize?: number;
  nNeighbors?: number;
  weight?: keyof typeof WEIGHT_FUNCTIONS;
  metric?: MetricName;
  metricParams?: MetricParams;
}
export class KNeighborBase implements KNeighborParams {
  algorithm?: NeighborMethodType;
  leafSize?: number;
  nNeighbors?: number;
  weight?: "uniform" | "distance";

  protected y?: string[] | boolean[] | number[];
  protected neighborMethod?: NeighborhoodMethod;
  protected weightFunction?: WeightFunction;

  constructor(parmas: KNeighborParams = {}) {
    const { algorithm = "ballTree", leafSize = 40, nNeighbors = 10, weight = "uniform", metric = 'l2', metricParams = {} } = parmas;
    this.algorithm = algorithm;
    this.neighborMethod = NeighborFactory.getNeighborMethod(algorithm, { leafSize, metric, metricParams });
    this.leafSize = leafSize;
    this.weight = weight;
    this.nNeighbors = nNeighbors;
    this.weightFunction = WEIGHT_FUNCTIONS[this.weight];
  }
  public async fit(xData: number[][] | Tensor2D, yData: string[] | boolean[] | number[] | Tensor1D): Promise<void> {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const yArray = checkJSArray(yData, 'any', 1) as number[] | boolean[] | string[];
    this.neighborMethod.fit(xArray);
    this.y = yArray;
  }
  public async query(xData: number[][] | Tensor2D): Promise<{ indices: number[][], distances?: number[][] }> {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    return this.neighborMethod.query(xArray, this.nNeighbors, true);
  }
  public async toObject(): Promise<Record<string, any>> {
    // const xArray = checkArray
    const modelParams: Record<string, any> = {};
    modelParams.algorithm = this.algorithm;
    modelParams.leafSize = this.leafSize;
    modelParams.weight = this.weight;
    modelParams.nNeighbors = this.nNeighbors;
    modelParams.neighborMethodParams = await this.neighborMethod.toObject();
    modelParams.y = this.y;
    return modelParams;
  }
  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    const {
      algorithm,
      leafSize,
      weight,
      nNeighbors,
      neighborMethodParams,
      y
    } = modelParams;
    this.algorithm = algorithm;
    this.neighborMethod = NeighborFactory.getNeighborMethod(algorithm);
    this.leafSize = leafSize;
    this.weight = weight;
    this.nNeighbors = nNeighbors;
    this.weightFunction = WEIGHT_FUNCTIONS[this.weight];
    this.y = y;
    await this.neighborMethod.fromObject(neighborMethodParams);
  }
}
