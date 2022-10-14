import { Tensor, RecursiveArray, tidy, dispose, Tensor2D } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { OneHotEncoder } from '../preprocess/encoder';
import { OneHotDropTypes } from '../preprocess/encoder';

export type FeatureInputType = Tensor | RecursiveArray<number>;
export type LabelInputType = Tensor | RecursiveArray<number>;

export type ClassMap = {
  [ key: string ]: number
}

export abstract class BaseEstimator {
  public estimatorType: string;
  public nFeature: number;
  constructor() {
    this.nFeature = 0;
  }
  public isClassifier(): boolean {
    return this.estimatorType === 'classifier';
  }
  /**
   * Check if input feature match the required feature size. If reset is true,
   * reset nFeature = x.shape[1]
   *
   * @param x Input feature of shape (nSample, nFeatures).
   * @param reset if true, the `nFeatures` attribute is set to `x.shape[1]`.
   */
  public checkAndSetNFeatures(x: Tensor | number[][], reset: boolean): void {
    if (x instanceof Tensor) {
      if (x?.shape?.length !== 2) {
        throw new TypeError('x should be 2D tensor');
      }
      const featureCount = x.shape[1];
      if (!reset && featureCount !== this.nFeature) {
        throw new TypeError(`X has ${featureCount} features, but is expected to have ${this.nFeature} features as input.`);
      }
      if (reset || !this.nFeature) {
        this.nFeature = x.shape[1];
      }
    } else {
      if (x instanceof Array && x[0] && x[0] instanceof Array) {
        const featureCount = x[0].length;
        if (!reset && featureCount !== this.nFeature) {
          throw new TypeError(`X has ${featureCount} features, but is expected to have ${this.nFeature} features as input.`);
        }
        if (reset || !this.nFeature) {
          this.nFeature = featureCount;
        }
        return;
      }
      throw new TypeError('Invalid input features');
    }
  }
  /**
   * Clean all tensor properties for memory release.
   */
  public clean(): void {
    for (const key in this) {
      const param = this[key];
      if (param instanceof Tensor) {
        dispose(param);
      }
    }
  }
  /**
   * Reset given parameter. If type of parameter is tensor, previous value
   * will be disposed.
   * @param paramKey parameter name
   * @param val value to be set
   */
  public cleanAndSetParam(paramKey: Extract<keyof this, string>, val: this[Extract<keyof this, string>]): void {
    if (paramKey in this) {
      const param = this[paramKey];
      if (param instanceof Tensor) {
        dispose(param);
      }
    }
    this[paramKey] = val;
  }
}

export class BaseClustering extends BaseEstimator {
  constructor() {
    super();
    this.estimatorType = 'clustering';
  }
  /**
   * Validate the input of clustering task
   * @param x input data
   * @param reset if reset the feature size
   * @returns tensor of input data
   */
  public validateData(x: FeatureInputType, reset: boolean): Tensor {
    return tidy(() => {
      const xTensor = checkArray(x, 'float32', 2) as Tensor2D;
      this.checkAndSetNFeatures(xTensor, reset);
      return xTensor;
    });
  }
}

export class BaseClassifier {
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

  public async getPredClass(score: Tensor): Promise<Tensor> {
    return await this.classOneHotEncoder.decode(score);
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
}

export class BaseRegressor extends BaseEstimator {
  // TODO(sugarspectre): Add evaluation functions
  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const y_tensor = checkArray(y, 'float32', yDimension);
    const xCount = xTensor.shape[0];
    const yCount = y_tensor.shape[0];
    if (xCount != yCount) {
      throw new RangeError(
        'The size of the training set and the training labels must be the same.'
      );
    }
    return { x: xTensor, y: y_tensor };
  }
}

// export class BaseEnsemble extends BaseEstimator {
//   public nEstimators: number;
//   public estimatorParams:
// }


export class BaseEstimater {
  // TODO(sugarspectre): Add evaluation functions
  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const y_tensor = checkArray(y, 'float32', yDimension);
    const xCount = xTensor.shape[0];
    const yCount = y_tensor.shape[0];
    if (xCount != yCount) {
      throw new RangeError(
        'The size of the training set and the training labels must be the same.'
      );
    }
    return { x: xTensor, y: y_tensor };
  }
}
