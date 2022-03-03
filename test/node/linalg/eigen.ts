import { eigenSolve } from '../../../src/linalg/eigen';
import * as fs from 'fs';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { assert } from 'chai';

const matrix = tf.tensor2d([
  [ 1, 2, 5, 10 ],
  [ 2, 6, 7, 5 ],
  [ 5, 7, 9, 6 ],
  [ 10, 5, 6, 7 ]
]);


const rawdata = fs.readFileSync('./test/node/linalg/covdata.json');
const covArray = JSON.parse(rawdata.toString());
const cov = tf.tensor2d(covArray);

describe('EigenSolver', () => {

  it('spectrum decomposition', async () => {
    const [ d, q ] = await eigenSolve(matrix);
    const di = tf.diag(d);
    const recovM = tf.matMul(tf.matMul(q, di), tf.transpose(q));
    const equal = tensorEqual(recovM, matrix, 1e-3);
    assert.isTrue(equal);
  });

  it('spectrum decomposition (covaraince)', async () => {
    const [ d, q ] = await eigenSolve(cov, 1e-3, 100, true);
    const di = tf.diag(d);
    const recovM = tf.matMul(tf.matMul(q, di), tf.transpose(q));
    const equal = tensorEqual(recovM, cov, 1e-2);
    assert.isTrue(equal);
  });
});
