import { Tensor, tensor, RecursiveArray, equal, sum, div } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

export class BaseClassifier {
  public estimatorType = 'classifier';
  public accuracyScore( yTrue: Tensor, yPred: Tensor ): number {
    const yTrueCount = yTrue.shape[0];
    const yPredCount = yPred.shape[0];
    if (yTrueCount != yPredCount) {
      throw new Error('Shape of yTrue should match shape of yPred');
    }
    // TODO(sugarspectre): Accuaracy score computation
    const score = div(sum(equal(yPred, yTrue)), yTrueCount).dataSync()[0];
    return score;
  }

  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | string[] | number[], xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    let yTensor: Tensor;
    if (! (y instanceof Tensor)) {
      yTensor = tensor(y);
    }
    //if (yTensor.dtype == 'int32' | yTensor.dtype ==)
    //yTensor = checkArray(y, 'int32', yDimension);
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
