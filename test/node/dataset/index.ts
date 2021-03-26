import { assert, expect } from 'chai';
import { DatasetMeta, DatasetSize, DatasetType, makeDataset, Sample } from '../../../src/dataset';
import MNIST from '../../../src/dataset/mnist';
import 'mocha';

class TestDatasetMeta implements DatasetMeta {
  type: DatasetType = DatasetType.Image;
  size: DatasetSize = {
    train: 3,
    test: 3
  };
  labelMap: { 1: '1' }
}

describe('Dataset', () => {
  it('should make a dataset', async () => {
    const sample: Sample<number> = {
      data: 1,
      label: 1
    }
    const trainSamples: Array<Sample> = [sample, sample, sample];
    const testSamples: Array<Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  
    expect(await dataset.getDatasetMeta()).to.eql(meta);
    expect(await dataset.train.next()).to.eql(sample);
    expect(await dataset.test.next()).to.eql(sample);
    expect(await dataset.train.nextBatch(3)).to.eql([sample, sample]);
    expect(await dataset.test.nextBatch(1)).to.eql([sample]);
  });

  it('should make a mnist dataset', async () => {
    const mnist = await MNIST.getMNIST();
    const meta = await mnist.getDatasetMeta();  

    const labelMap = {
      '0': '0',
      '1': '1',
      '2': '2',
      '3': '3',
      '4': '4',
      '5': '5',
      '6': '6',
      '7': '7',
      '8': '8',
      '9': '9'
    }

    expect(meta).to.eql({
      type: DatasetType.Image,
      size: { test: 60000, train: 10000 },
      dimension: { x: 28, y: 28, z: 1 },
      labelMap
    });
  });

  it('should read a zero batch', async () => {
    const sample: Sample<number> = {
      data: 1,
      label: 1
    }
    const trainSamples: Array<Sample> = [sample, sample, sample];
    const testSamples: Array<Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  
    expect(await dataset.train.nextBatch(0)).to.eql([]);
  });

  it('should read a whole batch', async () => {
    const sample: Sample<number> = {
      data: 1,
      label: 1
    }
    const trainSamples: Array<Sample> = [sample, sample, sample];
    const testSamples: Array<Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  
    expect(await dataset.train.nextBatch(-1)).to.eql(trainSamples);
  });
})
