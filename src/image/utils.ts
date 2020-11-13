import * as tf from '@tensorflow/tfjs-core';

export function stdCalc(data: tf.Tensor3D): number {

  return tf.moments(data).variance.sqrt().round().arraySync() as number;
}
