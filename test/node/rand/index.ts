import { beta } from '../../../src/rand';
import { expect } from 'chai';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import 'mocha';

describe('Beta Random test',
  () => { 
  it('should return tensor wiht one value', () => {
    const result = beta(1, 1);

    expect(result.shape).to.eql([1]);
    expect(result.dtype).to.eql('float32');
    expect(result.dataSync()).to.eql(new Float32Array(1).fill(1));
  });

  it('should return tensor wiht multi value', () => {
    const result = beta([1, 1, 1, 1], [1, 1, 1, 1], [2, 2]);

    expect(result.shape).to.eql([2, 2]);
    expect(result.dtype).to.eql('float32');
    expect(result.dataSync()).to.eql(new Float32Array(4).fill(1));
  });
});

