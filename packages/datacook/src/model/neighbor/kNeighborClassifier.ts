import { dispose, memory, mul, RecursiveArray, reshape, sum, Tensor, Tensor1D, tensor2d, Tensor2D, tidy } from "@tensorflow/tfjs-core";
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

  public async initClasses(y: Tensor | number[] | string[] | boolean[], drop: OneHotDropTypes = 'none'): Promise<void> {
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
    const ySlice: (string | boolean | number)[] = [];
    for (let i = 0; i < indices.length; i++) {
      for (let j = 0; j < indices[i].length; j++) {
        ySlice.push(this.y[indices[i][j]]);
      }
    }
    // const ySlice = indices.map((inds) => inds.map((d) => this.y[d]));
    // const nnLabels = reshape(tensor2d(ySlice), [ nSamples * nNeighbors ]);
    const weights = this.weightFunction(distTensor);
    const labelOneHot = await this.classOneHotEncoder.encode(ySlice as string[] | boolean[] | number[]);
    const proba = tidy(() => sum(mul(reshape(labelOneHot, [ nSamples, nNeighbors, -1 ]), reshape(weights, [ nSamples, nNeighbors, 1 ])), 1) as Tensor2D);
    dispose([ weights, labelOneHot, distTensor ]);
    return proba;
  }

  public async fit(xData: number[][] | Tensor2D, yData: number[] | string[] | boolean[] | Tensor1D): Promise<void> {
    await this.initClasses(yData);
    await super.fit(xData, yData);
  }
  public async predict(xData: number[][] | Tensor2D): Promise<Tensor> {
    const proba = await this.predictProba(xData);
    const predClass = await this.classOneHotEncoder.decode(proba);
    dispose(proba);
    return predClass;
  }
  public async toObject(): Promise<Record<string, any>> {
    const modelParams = await super.toObject();
    const categories = this.classOneHotEncoder.categories.arraySync();
    modelParams.categories = categories;
    return modelParams;
  }
  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    const {
      categories
    } = modelParams;
    await super.fromObject(modelParams);
    await this.initClasses(categories);
  }
  public async toJson(): Promise<string> {
    const modelParams = await this.toObject();
    return JSON.stringify(modelParams);
  }
  public async fromJson(modelJson: string): Promise<void> {
    const modelParams = JSON.parse(modelJson);
    await this.fromObject(modelParams);
  }
}
