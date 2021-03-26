import { split, npy, shuffle, seed } from '../../../src/generic';
import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import 'mocha';
import * as fs from 'fs';

describe('Generic Split test',
  () => { 
    it('should split data into train & test', () => { 
      /**
       * [
       *   [0, 1],
       *   [2, 3],
       *   [4, 5],
       *   [6, 7],
       *   [8, 9]
       * ]
       */
      const X = tf.range(0, 10).reshape([5, 2]);
      
      const y = tf.range(0, 5);

      const [X_train, X_test, y_train, y_test] = split([X, y]); 
      
      const actX_test = tf.range(6, 10).reshape([2, 2]);
      const actX_train = tf.range(0, 6).reshape([3, 2]);
      
      const acty_train = tf.range(0, 3).reshape([3]);
      const acty_test = tf.range(3, 5).reshape([2]);

      expect(actX_test.dataSync()).to.eql(X_test.dataSync());
      expect(actX_train.dataSync()).to.eql(X_train.dataSync());

      expect(acty_train.dataSync()).to.eql(y_train.dataSync());
      expect(acty_test.dataSync()).to.eql(y_test.dataSync());
  }); 

  it('should throws if tensors length equals zero ', () => { 
    expect(() => split([])).to.throw('inputs should not have length of zero');
  }); 

  it('should throws if inputs have different first dimension', () => { 
    const t1 = tf.ones([10, 28, 28, 3]);
    const t2 = tf.ones([8]);
    expect(() => split([t1, t2])).to.throw('inputs should have the same length');
  }); 
});

describe('Generic Parse test', () => { 
  it('should read npy', () => {
    // ones npy 100*100
    const test = fs.readFileSync('test/node/generic/artifacts/test.npy').buffer;
    const data = npy.parse(test);
    
    expect(data.shape).to.eql([100, 100]);
    expect(data.dtype).to.eql('float32');
    expect(data.data).to.eql(new Float32Array(100*100).fill(1));
  });

  it('should read npy to tfjs tensor', () => {
    const test = fs.readFileSync('test/node/generic/artifacts/test.npy').buffer;
    const tensor = npy.parse2Tensor(test);

    expect(tensor.shape).to.eql([100, 100]);
    expect(tensor.dtype).to.eql('float32');
    expect(tensor.dataSync()).to.eql(new Float32Array(100*100).fill(1));
  });
});


describe('Random shuffle', () => {
  const arr = new Array(10).fill(1).map((_, idx) => idx);
  seed('test');
  shuffle(arr);
  const expectedArr = [5, 9, 0, 4, 1, 6, 2, 7, 3, 8];

  expect(arr).eql(expectedArr);
})

describe('Seeding random', () => {
  seed('test');
  const arr = new Array(10).fill(1).map((_) => Math.random());
  const expectedArr = [
    0.8722025543160253,
    0.4023928518604753,
    0.9647289658507073,
    0.30479896375101545,
    0.3521069009157321,
    0.2734533903544762,
    0.4635571187776387,
    0.10034856760950056,
    0.7247513588372084,
    0.4236748288641446
  ];

  seed('test1');
  const arr1 = new Array(10).fill(1).map((_) => Math.random());
  const expectedArr1 = [
    0.4140663576925043,
    0.3835913058226645,
    0.32223050456935015,
    0.5744860625750371,
    0.720112122139951,
    0.6619058590665984,
    0.9665898051176159,
    0.4101568136478809,
    0.8431809602909894,
    0.8021843641109205
  ];

  expect(arr).eql(expectedArr);
  expect(arr1).eql(expectedArr1);
})

