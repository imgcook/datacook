import { Tensor, tensor, RecursiveArray, unique, cast, oneHot } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

export type ClassMap = {
  [ key: string ]: number
}

export class BaseClassifier {
  public estimatorType = 'classifier';
  public classes: Tensor;
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
    const yOneHot = oneHot(cast(tensor(yInd), 'int32'), nClass);
    return yOneHot;
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
        'the size of training set and training labels must be the same.'
      );
    }
    return { 'x': xTensor, 'y': yTensor };
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
    return { 'x': xTensor, 'y': y_tensor };
  }
}
