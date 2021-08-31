import { svd } from '../../../src/linalg/svd';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';
import 'mocha';
import { disableDeprecationWarnings, max } from '@tensorflow/tfjs-core';

const matrix = tf.tensor2d([
  [1, 5, 7, 6, 1],
  [2, 1, 10, 4, 4],
  [3, 6, 7, 5, 2]
]);

describe('SVDSolver', () => {
    
    it('svd decomposition', () => {
      const [ u, d, v ] = svd(matrix);
      u.print();
      v.print();
      d.print();
      const recovM = tf.matMul(tf.matMul(u, d), tf.transpose(v));
      recovM.print();
    })
  });

