import * as tf from '@tensorflow/tfjs-core';

export function stdCalc(data: tf.Tensor3D): number {
  return tf.round(tf.sqrt(tf.moments(data).variance)).arraySync() as number;
}
