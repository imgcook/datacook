import { add, any, divNoNan, equal, fill, lessEqual, mul, neg, sum, Tensor1D, Tensor2D, tidy, transpose } from "@tensorflow/tfjs-core";
import { checkArray, checkJSArray } from "../../utils/validation";
import { BallTree } from "./ballTree";
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
      const weights = divNoNan(inverseDist, invSum);
      return transpose(add(mul(transpose(weights), neg(hasZero)), isZero)) as Tensor2D;
    });
  }
};

export const NEIGHBOR_METHODS = {
  ballTree: new BallTree()
};

export interface KNeighborParams {
  algorithm?: keyof typeof NEIGHBOR_METHODS;
  leafSize?: number;
  nNeighbors?: number;
  weight?: keyof typeof WEIGHT_FUNCTIONS;
}
export class KNeighborBase implements KNeighborParams {
  algorithm?: keyof typeof NEIGHBOR_METHODS;
  leafSize?: number;
  nNeighbors?: number;
  weight?: "uniform" | "distance";

  protected y?: Tensor1D;
  protected neighborMethod?: NeighborhoodMethod;
  protected weightFunction?: WeightFunction;

  constructor(parmas: KNeighborParams = {}) {
    const { algorithm = "ballTree", leafSize = 40, nNeighbors = 10, weight = "uniform" } = parmas;
    this.algorithm = algorithm;
    this.neighborMethod = NEIGHBOR_METHODS[this.algorithm];
    this.leafSize = leafSize;
    this.weight = weight;
    this.nNeighbors = nNeighbors;
    this.weightFunction = WEIGHT_FUNCTIONS[this.weight];
  }
  public async fit(xData: number[][] | Tensor2D, yData: Tensor1D): Promise<void> {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const yTensor = checkArray(yData, 'any', 1) as Tensor1D;
    this.neighborMethod.fit(xArray, { leafSize: this.leafSize });
    this.y = yTensor;
  }
  public async query(xData: number[][] | Tensor2D): Promise<{ indices: number[][], distances?: number[][] }> {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    return this.neighborMethod.query(xArray, this.nNeighbors, true);
  }
}
