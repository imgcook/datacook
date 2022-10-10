import { gather, mul, RecursiveArray, reshape, sum, Tensor, tensor2d, Tensor2D } from "@tensorflow/tfjs-core";
import { OneHotDropTypes, OneHotEncoder } from "../../preprocess/encoder";
import { checkArray, checkJSArray } from "../../utils/validation";
import { BaseClassifier, ClassMap } from "../base";
import { KNeighborBase } from "./kneighborBase";

export class KNeighborClassifier extends KNeighborBase implements BaseClassifier {
  public estimatorType = 'classifier';
  public classOneHotEncoder: OneHotEncoder;
  public classMap: ClassMap;

  // get label one-hot expression
  public async getLabelOneHot(y: Tensor): Promise<Tensor> {
    return await this.classOneHotEncoder.encode(y);
  }

  public async initClasses(y: Tensor | number[] | string[], drop: OneHotDropTypes = 'none'): Promise<void> {
    this.classOneHotEncoder = new OneHotEncoder({ drop });
    await this.classOneHotEncoder.init(y);
  }

  public classes(): Tensor {
    return this.classOneHotEncoder?.categories;
  }

  public isBinaryClassification(): boolean {
    return this.classes().shape[0] === 2;
  }

  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const yTensor = checkArray(y, 'any', yDimension);
    const xCount = xTensor.shape[0];
    const yCount = yTensor.shape[0];
    if (xCount != yCount) {
      throw new RangeError(
        'The size of training set and training labels must be the same.'
      );
    }
    return { x: xTensor, y: yTensor };
  }

  public async getPredClass(score: Tensor): Promise<Tensor> {
    return await this.classOneHotEncoder.decode(score);
  }
  /**
   * Compute class probability
   * @param xData input features
   * @returns tensor of probabilities
   */
  public async predictProba(xData: number[][] | Tensor2D): Promise<Tensor2D> {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const { distances, indices } = await this.query(xArray);
    const distTensor = tensor2d(distances);
    const [ nSamples, nNeighbors ] = distTensor.shape;
    const nnLabels = gather(this.y, indices);
    const weights = this.weightFunction(distTensor);
    const labelOneHot = await this.classOneHotEncoder.encode(nnLabels);
    const proba = sum(mul(reshape(labelOneHot, [ nSamples, nNeighbors, -1 ]), reshape(weights, [ nSamples, nNeighbors, 1 ])), 1) as Tensor2D;
    return proba;
  }
  public async predict(xData: number[][] | Tensor2D): Promise<Tensor> {
    const proba = await this.predictProba(xData);
    return await this.classOneHotEncoder.decode(proba);
  }
}
