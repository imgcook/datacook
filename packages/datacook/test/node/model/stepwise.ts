import * as tf from '@tensorflow/tfjs-core';
import { stepwiseLinearRegression } from '../../../src/model/stat/stepwise';

const nData = 100;
const cases = tf.mul(tf.randomNormal([ nData, 5 ]), [ 1, 10, 100, 2, 3 ]);
const weight = tf.tensor([ 2, 3, 1, -4, 6 ]);
const noise = tf.mul(tf.randomNormal([ nData ]), 2);
const y = tf.add(tf.add(tf.sum(tf.mul(cases, weight), 1), 10), noise);

const treesGrith = [ 8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11.0, 11.0, 11.1,
  11.2, 11.3, 11.4, 11.4, 11.7, 12.0, 12.9, 12.9, 13.3, 13.7, 13.8, 14.0, 14.2, 14.5,
  16.0, 16.3, 17.3, 17.5, 17.9, 18.0, 18.0, 20.6 ];

const treesHeight = [ 70, 65, 63, 72, 81, 83, 66, 75, 80, 75, 79, 76, 76,
  69, 75, 74, 85, 86, 71, 64, 78, 80, 74, 72, 77, 81, 82, 80, 80, 80, 87 ];

const treesVolumn = [ 10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6,
  19.9, 24.2, 21.0, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9, 34.5, 31.7, 36.3,
  38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51.0, 77.0 ];

describe('Linear Regression', () => {

  it('train simple dataset', async () => {
    const lm = await stepwiseLinearRegression(cases, y, [ 'x1', 'x2', 'x3', 'x4', 'x5' ]);
    const summary = lm.summary();
    console.log(summary);
  });

  it('train on tree dataset', async () => {
    const treeFeatureTensor = tf.transpose(tf.tensor2d([ treesHeight, treesVolumn ]));
    const lm = await stepwiseLinearRegression(treeFeatureTensor, treesGrith, [ 'height', 'volumn' ]);
    lm.printSummary();
  });
});
