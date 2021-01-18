import { split, npy } from '../../../src/generic';
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

