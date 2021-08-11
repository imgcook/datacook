import { eigenSolve } from '../../../src/linalg/eigen';
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { max } from '@tensorflow/tfjs-core';

const matrix = tf.tensor2d([
    [ 1, 2, 3, 10 ],
    [ 2, 6, 7, 5 ],
    [ 3, 7, 9, 6 ],
    [ 10, 5, 6, 7]
]);

describe('EigenSolver', () => {
    
    it('solve matrix', () => {
        const [ q, d ] = eigenSolve(matrix);
        q.print();
        d.print();
        const recovM = tf.matMul(tf.matMul(q, d), tf.transpose(q));
        const dm = tf.matMul(tf.matMul(tf.transpose(q), matrix), q);
        dm.print();
        recovM.print();
    })
})