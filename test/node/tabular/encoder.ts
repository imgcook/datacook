import { LabelEncoder, OneHotEncoder } from '../../../src/tabular/encoder';
import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';


describe('Encodings', function () {

  describe('LabelEncoder', function () {

    it('test the label encoding on array', async function () {
      const data = [ 'dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat' ];
      const encode = new LabelEncoder();
      const fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(await encode.fit(data), fit_data);
      assert.deepEqual(await encode.transform([ 'dog', 'man' ]), [ 0, 2 ]);
    });

    it('test the label encoding on Tensor', async function () {
      const data = [ 'dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat' ];
      const tensor_data = tf.tensor(data);
      const encode = new LabelEncoder();
      const fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(await encode.fit(tensor_data), fit_data);
      assert.deepEqual(await encode.transform(tf.tensor([ 'dog', 'man' ])), [ 0, 2 ]);
    });

  });


  describe('OneHotEncoder', function () {

    it('test one-hot encoding on array', async function () {
      const data = [ 'dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat' ];
      const encode = new OneHotEncoder();
      const fit_data = [
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, 1 ],
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, 1 ],
        [ 0, 0, 1 ],
        [ 0, 1, 0 ]
      ];
      const transform_data = [ [ 0, 0, 1 ], [ 0, 1, 0 ] ];

      assert.deepEqual(await encode.fit(data), fit_data);
      assert.deepEqual(await encode.transform([ 'man', 'cat' ]), transform_data);
    });

    it('test onehotencoding on Tensor', async function () {
      const data = [ 'dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat' ];
      const tensor_data = tf.tensor(data);
      const encode = new OneHotEncoder();
      const fit_data = [
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, 1 ],
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, 1 ],
        [ 0, 0, 1 ],
        [ 0, 1, 0 ]
      ];
      const transform_data = [ [ 0, 0, 1 ], [ 0, 1, 0 ] ];

      assert.deepEqual(await encode.fit(tensor_data), fit_data);
      assert.deepEqual(await encode.transform(tf.tensor([ 'man', 'cat' ])), transform_data);
    });
  });
});
