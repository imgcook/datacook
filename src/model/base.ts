import { Tensor, tensor, RecursiveArray } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { OneHotEncoder } from '../preprocess';
import { OneHotDropTypes } from '../preprocess/encoder';

export type ClassMap = {
  [ key: string ]: number
}

export class BaseClassifier {
  public estimatorType = 'classifier';
  public classOneHotEncoder: OneHotEncoder;
  public classMap: ClassMap;

  public score( x: Tensor, y: Tensor ): Tensor {
    const xCount = x.shape[0];
    const yCount = y.shape[0];
    if (xCount == 0 || yCount == 0) {
      throw new Error('inputs should not have length of zero');
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

  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const yTensor = checkArray(y, 'any', yDimension);
    const xCount = xTensor.shape[0];
    const yCount = yTensor.shape[0];
    if (xCount != yCount) {
      throw new RangeError(
        'the size of training set and training labels must be the same.'
      );
    }
    return { x: xTensor, y: yTensor };
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
        'the size of the training set and the training labels must be the same.'
      );
    }
    return { x: xTensor, y: y_tensor };
  }
}
