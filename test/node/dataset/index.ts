import { expect } from 'chai';
import { ArrayDatasetImpl, makeTransformDataset, Types } from '../../../src/dataset';
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

describe('Dataset', () => {
  it('should make a dataset', async () => {
    const sample: Types.Sample<number> = {
      data: 1,
      label: 1
    };
    const trainSamples: Array<Types.Sample> = [ sample, sample, sample ];

    const dataset = new ArrayDatasetImpl(trainSamples);

    expect(await dataset.next()).to.eql(sample);
    expect(await dataset.nextBatch(3)).to.eql([ sample, sample ]);
    await dataset.seek(0);
    expect(await dataset.nextBatch(-1)).to.eql([ sample, sample, sample ]);
    expect(await dataset.next()).to.eql(undefined);
    expect(await dataset.nextBatch(1)).to.eql([]);
  });

  it('should make a transform', async () => {
    const sample: Types.Sample<number> = {
      data: 1,
      label: 1
    };
    const transformedSample: Types.Sample<number> = {
      data: 2,
      label: 2
    };
    const trainSamples: Array<Types.Sample> = [ sample, sample, sample ];

    const dataset = new ArrayDatasetImpl(trainSamples);
    const transformed = makeTransformDataset(dataset, async (sample: Types.Sample): Promise<Types.Sample> => {
      return {
        data: sample.data + 1,
        label: sample.data + 1
      };
    });
    expect(await transformed.next()).to.eql(transformedSample);
    expect(await transformed.nextBatch(3)).to.eql([ transformedSample, transformedSample ]);
    await transformed.seek(0);
    expect(await transformed.nextBatch(-1)).to.eql([ transformedSample, transformedSample, transformedSample ]);
    expect(await transformed.next()).to.eql(undefined);
    expect(await transformed.nextBatch(1)).to.eql([]);
  });

  it('should iter a dataset after shuffle', async () => {
    const sampleMaker = (num: number) => {
      return {
        data: num,
        label: num
      }
    };
    const trainSamples: Array<Types.Sample> = [sampleMaker(0), sampleMaker(1), sampleMaker(2)];
  
    const dataset = new ArrayDatasetImpl(trainSamples);

    seed('test');
    dataset.shuffle();

    const trainData = (await dataset.nextBatch(3)).map(it => it.data);

    expect(trainData).to.eql([1, 0, 2]);
  });

  it('should read a zero batch', async () => {
    const sample: Types.Sample<number> = {
      data: 1,
      label: 1
    }
    const trainSamples: Array<Types.Sample> = [sample, sample, sample];
    const dataset = new ArrayDatasetImpl(trainSamples);
  
    expect(await dataset.nextBatch(0)).to.eql([]);
  });

  it('should throw an error', async () => {
    const sample: Types.Sample<number> = {
      data: 1,
      label: 1
    };
    const trainSamples: Array<Types.Sample> = [ sample, sample, sample ];
  
    const dataset = new ArrayDatasetImpl(trainSamples);

    await expectThrowsAsync(() => dataset.nextBatch(-2));
  });
});

