import * as tf from '@tensorflow/tfjs-core';
import { assert } from 'chai';
import 'mocha';
import { OneHotEncoder } from '../../../src/preprocess';

const x = [ 'tree', 'apple', 'banana', 'tree', 'apple', 'banana' ];
const xEncode = tf.tensor([
  [ 1, 0, 0 ],
  [ 0, 1, 0 ],
  [ 0, 0, 1 ],
  [ 1, 0, 0 ],
  [ 0, 1, 0 ],
  [ 0, 0, 1 ]
]);
const xEncodeDrop = tf.tensor([
  [ 0, 0 ],
  [ 1, 0 ],
  [ 0, 1 ],
  [ 0, 0 ],
  [ 1, 0 ],
  [ 0, 1 ]
]);
const bx = [ 'tree', 'apple', 'tree', 'apple' ];
const bxEncodeDrop = tf.tensor([ 0, 1, 0, 1 ]);

describe('OneHot Encoder', () => {
  it('encode', async () => {
    const encoder = new OneHotEncoder();
    await encoder.init(x);
    const xOneHot = await encoder.encode(x);
    assert.deepEqual(xOneHot.dataSync(), xEncode.dataSync());
  });
  it('decode', async () => {
    const encoder = new OneHotEncoder();
    await encoder.init(x);
    const xCate = await encoder.decode(xEncode);
    assert.deepEqual(xCate.dataSync() as any, x);
  });
  it('encode drop first', async () => {
    const encoder = new OneHotEncoder({ drop: 'first' });
    await encoder.init(x);
    const xOneHot = await encoder.encode(x);
    assert.deepEqual(xOneHot.dataSync(), xEncodeDrop.dataSync());
  });
  it('encode binary only', async () => {
    const encoder = new OneHotEncoder({ drop: 'binary-only' });
    await encoder.init(bx);
    const bxOneHot = await encoder.encode(bx);
    assert.deepEqual(bxOneHot.dataSync(), bxEncodeDrop.dataSync());
  });
  it('decode binary only', async () => {
    const encoder = new OneHotEncoder({ drop: 'binary-only' });
    await encoder.init(bx);
    const bxCate = await encoder.decode(bxEncodeDrop);
    assert.deepEqual(bxCate.dataSync() as any, bx);
  });
});
