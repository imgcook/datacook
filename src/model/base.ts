import { Tensor, tensor, RecursiveArray } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

export class BaseClassifier {
  public estimatorType = 'classifier';
  public score( X: Tensor, y: Tensor ): Tensor {
    const n_X = X.shape[0];
    const n_y = y.shape[0];
    if (n_X == 0 || n_y == 0) {
      throw new Error('inputs should not have length of zero');
    }
    // TODO(sugarspectre):
    // Accuaracy score computation
    const score = tensor([ 0 ]);
    return score;
  }

  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const yTensor = checkArray(y, 'int32', yDimension);
    const n_X = xTensor.shape[0];
    const n_y = yTensor.shape[0];
    if (n_X != n_y) {
      throw new RangeError(
        'the size of training set and training labels must be the same.'
      );
    }
    return { 'x': xTensor, 'y': yTensor };
  }
}


export class BaseEstimater {
  // TODO(sugarspectre):
  // Add evaluation functions
  public validateData(x: Tensor | RecursiveArray<number>, y: Tensor | RecursiveArray<number>, xDimension = 2, yDimension = 1): { x: Tensor, y: Tensor } {
    const xTensor = checkArray(x, 'float32', xDimension);
    const y_tensor = checkArray(y, 'float32', yDimension);
    const n_X = xTensor.shape[0];
    const n_y = y_tensor.shape[0];
    if (n_X != n_y) {
      throw new RangeError(
        'the size of the training set and the training labels must be the same.'
      );
    }
    return { 'x': xTensor, 'y': y_tensor };
  }
}
