import { expect } from 'chai';
import { makeDataset, Types } from '../../../src/dataset';
import MNIST from '../../../src/dataset/mnist';
import 'mocha';
import { seed } from '../../../src/generic';

const expectThrowsAsync = async (method: any, errorMessage?: string) => {
  let error = null
  try {
    await method()
  }
  catch (err) {
    error = err
  }
  expect(error).to.be.an('Error')
  if (errorMessage) {
    expect(error.message).to.equal(errorMessage)
  }
}

class TestDatasetMeta implements Types.DatasetMeta {
  type: Types.DatasetType = Types.DatasetType.Image;
  size: Types.DatasetSize = {
    train: 3,
    test: 3
  };
  labelMap: { 1: '1' }
}

describe('Dataset', () => {
  it('should make a dataset', async () => {
    const sample: Types.Sample<number> = {
      data: 1,
    }
    const trainSamples: Array<Types.Sample> = [sample, sample, sample];
    const testSamples: Array<Types.Sample> = [sample, sample, sample];
  
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

  it('should iter a dataset after shuffle', async () => {
    const sampleMaker = (num: number) => {
      return {
        data: num,
      }
    }
    const trainSamples: Array<Types.Sample> = [sampleMaker(0), sampleMaker(1), sampleMaker(2)];
    const testSamples: Array<Types.Sample> = [sampleMaker(3), sampleMaker(4), sampleMaker(5)];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);

    seed('test');
    dataset.shuffle();

    const trainData = (await dataset.train.nextBatch(3)).map(it => it.data);
    const testData = (await dataset.test.nextBatch(3)).map(it => it.data);
  
    expect(await dataset.getDatasetMeta()).to.eql(meta);
    expect(trainData).to.eql([1, 0, 2])
    expect(testData).to.eql([4, 3, 5])
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
      type: Types.DatasetType.Image,
      size: { test: 60000, train: 10000 },
      dimension: { x: 28, y: 28, z: 1 },
      labelMap
    });
  });

  it('should read a zero batch', async () => {
    const sample: Types.Sample<number> = {
      data: 1
    }
    const trainSamples: Array<Types.Sample> = [sample, sample, sample];
    const testSamples: Array<Types.Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  
    expect(await dataset.train.nextBatch(0)).to.eql([]);
  });

  it('should read a whole batch', async () => {
    const sample: Types.Sample<number> = {
      data: 1
    }
    const trainSamples: Array<Types.Sample> = [sample, sample, sample];
    const testSamples: Array<Types.Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  
    expect(await dataset.train.nextBatch(-1)).to.eql(trainSamples);
  });

  it('should throw an error', async () => {
    const sample: Types.Sample<number> = {
      data: 1
    }
    const trainSamples: Array<Types.Sample> = [sample, sample, sample];
    const testSamples: Array<Types.Sample> = [sample, sample, sample];
  
    const meta = new TestDatasetMeta();
  
    const dataset = makeDataset({
      trainData: trainSamples,
      testData: testSamples,
    }, meta);
  

    const expectedError = new RangeError(`Batch size should be larger than -1 but -2 is present`);
    await expectThrowsAsync(() => dataset.train.nextBatch(-2));
  })
})
