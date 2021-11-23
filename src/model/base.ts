<<<<<<< HEAD
import { Tensor, tensor, RecursiveArray, unique, cast, oneHot, argMax, reshape, stack, slice, greater } from '@tensorflow/tfjs-core';
=======
import { Tensor, RecursiveArray, tensor, tidy, dispose, Tensor2D } from '@tensorflow/tfjs-core';
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
import { checkArray } from '../utils/validation';
import { OneHotEncoder } from '../preprocess';
import { OneHotDropTypes } from '../preprocess/encoder';

export type FeatureInputType = Tensor | RecursiveArray<number>;
export type LabelInputType = Tensor | RecursiveArray<number>;

export type ClassMap = {
  [ key: string ]: number
}

export abstract class BaseEstimator {
  public estimatorType: string;
  public nFeature: number;

  /**
   * Check if input feature match the required feature size. If reset is true,
   * reset nFeature = x.shape[1]
   *
   * @param x Input feature of shape (nSample, nFeatures).
   * @param reset if true, the `nFeatures` attribute is set to `x.shape[1]`.
   */
  public checkAndSetNFeatures(x: Tensor, reset: boolean): void {
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

export type ClassMap = {
  [ key: string ]: number
}

export class BaseClassifier {
  public estimatorType = 'classifier';
<<<<<<< HEAD
  public classes: Tensor;
=======
  public classOneHotEncoder: OneHotEncoder;
>>>>>>> a204a8a299a83931037e6addd72fe7816251b3e4
  public classMap: ClassMap;

  public score( x: Tensor, y: Tensor ): Tensor {
    const xCount = x.shape[0];
    const yCount = y.shape[0];
    if (xCount == 0 || yCount == 0) {
      throw new Error('Inputs should not have length of zero');
    }
    // TODO(sugarspectre): Accuaracy score computation
    const score = tensor([ 0 ]);
    return score;
  }
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

  // update class map
  public updateClassMap(): void {
    if (this.classes){
      const classData = this.classes.dataSync();
      const classMap: ClassMap = {};
      for (let i = 0; i < classData.length; i++) {
        const key = classData[i];
        classMap[key] = i;
      }
      this.classMap = classMap;
    }
  }

  // get label one-hot expression
  public getLabelOneHot(y: Tensor): Tensor {
    const yData = y.dataSync();
    const nClass = this.classes.shape[0];
    const yInd = yData.map((d: number|string) => this.classMap[d]);
    //if (nClass > 2) {
    const yOneHot = oneHot(cast(tensor(yInd), 'int32'), nClass);
    return yOneHot;
    // } else {
    //   return cast(tensor(yInd), 'int32');
    // }
  }

  public getLabelOneHotBinary(y: Tensor): Tensor {
    const yData = y.dataSync();
    const yInd = yData.map((d: number|string) => this.classMap[d]);
    if (this.isBinaryClassification) {
      return cast(tensor(yInd), 'int32');
    } else {
      throw new Error('Can only except binary classification input');
    }
  }

  public initClasses(y: Tensor): void {
    const { values } = unique(y);
    this.classes = values;
    this.updateClassMap();
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

  public isBinaryClassification(): boolean {
    const nClass = this.classes.shape[0];
    return nClass == 2;
  }

  public getPredClassBinary(score: Tensor): Tensor {
    if (!this.isBinaryClassification()) {
      throw new Error ('Can only except binary classification input');
    }
    const classInd = greater(score, 0.5).dataSync();
    const classTensors: Tensor[] = [];
    classInd.forEach((i: number) => { return classTensors.push(slice(this.classes, [ i ], [ 1 ])); });
    const classVal = reshape(stack(classTensors), [ -1 ]);
    return classVal;

  }

  public getPredClass(score: Tensor): Tensor {
    const axisH = 1;
    // const classInd = (this.isBinaryClassification) ? greater(score, 0.5).dataSync() : argMax(score, axisH).dataSync();
    const classInd = argMax(score, axisH).dataSync();
    const classTensors: Tensor[] = [];
    classInd.forEach((i: number) => { return classTensors.push(slice(this.classes, [ i ], [ 1 ])); });
    const classVal = reshape(stack(classTensors), [ -1 ]);
    return classVal;
  }
}

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
