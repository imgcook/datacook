import { FactorAnalysis } from '../../../src/model/factor-analysis/index';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';

const matrix = tf.tensor2d([
  [ 1, 5, 7, 6, 1 ],
  [ 2, 1, 10, 4, 4 ],
  [ 3, 6, 7, 5, 2 ],
  [ 6, 7, 8, 9, 10 ],
  [ 8, 8, 9, 9, 5 ],
  [ 10, 2, 4, 5, 2 ]
]);

describe('Factor analysis', () => {
  it('', async () => {
    const fa = new FactorAnalysis({nComponent: -1});
    fa.fit(matrix);
  });

});