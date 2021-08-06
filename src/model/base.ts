import { Tensor, tensor } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';

export class BaseClassifier {
  public estimatorType = 'classifier';
  public score( X: Tensor, y: Tensor ): Tensor {
    const n_X = X.shape[0];
    const n_y = y.shape[0];
    if (n_X == 0 || n_y == 0) {
      throw new Error('inputs should not have length of zero');
    }
    // TODO:
    // Accuaracy score computation
    const score = tensor([ 0 ]);
    return score;
  }

  // eslint-disable-next-line @typescript-eslint/explicit-module-boundary-types
  public validateData(X: any, y: any) {
    const X_tensor = checkArray(X, 'float32', 2);
    const y_tensor = checkArray(y, 'int32', 1);
    const n_X = X_tensor.shape[0];
    const n_y = y_tensor.shape[0];
    if (n_X != n_y) {
      throw new RangeError(
        "the size of the training set and the training labels must be the same."
      );
    }
    return { 'X': X_tensor, 'y': y_tensor };
  }
}
