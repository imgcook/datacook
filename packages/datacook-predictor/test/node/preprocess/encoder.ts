import { assert } from 'chai';
import 'mocha';
import { OneHotEncoder, LabelEncoder } from '../../../src/preprocess/encoder';


const x = ['tree', 'apple', 'banana', 'tree', 'apple', 'banana'];
const xEncode = [
  [ 1, 0, 0 ],
  [ 0, 1, 0 ],
  [ 0, 0, 1 ],
  [ 1, 0, 0 ],
  [ 0, 1, 0 ],
  [ 0, 0, 1 ]];
const xLabelEncode = [ 0, 1, 2, 0, 1, 2 ];
const xEncodeDrop = [
  [ 0, 0 ],
  [ 1, 0 ],
  [ 0, 1 ],
  [ 0, 0 ],
  [ 1, 0 ],
  [ 0, 1 ],
];
const bx = [ 'tree', 'apple', 'tree', 'apple' ];
const bxEncodeDrop = [ 0, 1, 0, 1 ];

describe('OneHot Encoder', () => {
  it('encode', async () => {
    const encoder = new OneHotEncoder();
    await encoder.init(x);
    const xOneHot = await encoder.encode(x);
    assert.deepEqual(xOneHot, xEncode);
  });
  it('decode', async () => {
    const encoder = new OneHotEncoder();
    await encoder.init(x);
    const xCate = await encoder.decode(xEncode);
    assert.deepEqual(xCate, x);
  });
  it('encode drop first', async () => {
    const encoder = new OneHotEncoder({drop: 'first'});
    await encoder.init(x);
    const xOneHot = await encoder.encode(x);
    assert.deepEqual(xOneHot, xEncodeDrop);
  });
  it('encode binary only', async () => {
    const encoder = new OneHotEncoder({drop: 'binary-only'});
    await encoder.init(bx);
    const bxOneHot = await encoder.encode(bx);
    assert.deepEqual(bxOneHot, bxEncodeDrop);
  });
  it('decode binary only', async () => {
    const encoder = new OneHotEncoder({drop: 'binary-only'});
    await encoder.init(bx);
    const bxCate = await encoder.decode(bxEncodeDrop);
    assert.deepEqual(bxCate, bx);
  });
});

describe('Label Encoder', () => {
  it('encode', async () => {
    const encoder = new LabelEncoder();
    await encoder.init(x);
    const xEncode = await encoder.encode(x);
    assert.deepEqual(xEncode, xLabelEncode);
  });
  it('decode', async () => {
    const encoder = new LabelEncoder();
    await encoder.init(x);
    const xDecode = await encoder.decode(xLabelEncode);
    assert.deepEqual(x, xDecode);
  })
});
