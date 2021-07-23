import { expect } from 'chai';
import { Types } from '../../../src/dataset';
import { makeDatasetFromCsv } from '../../../src/dataset/format/csv';
import 'mocha';
import { Csv } from '../../../src/dataset/types';
import { csvDataWithHead, csvDataWithoutHead } from './data';

const sample1: Csv.Sample = {
  data: {
    A: '1',
    B: '2'
  },
  label: {
    C: '3'
  }
};

const sample2: Csv.Sample = {
  data: {
    A: '4',
    B: '5'
  },
  label: {
    C: '6'
  }
};

const sample3: Csv.Sample = {
  data: {
    A: '7',
    B: '8'
  },
  label: {
    C: '9'
  }
};

const sampleNoHead1: Csv.Sample = {
  data: {
    '0': '1',
    '1': '2'
  },
  label: {
    '2': '3'
  }
};

const sampleNoHead2: Csv.Sample = {
  data: {
    '0': '4',
    '1': '5'
  },
  label: {
    '2': '6'
  }
};

const sampleNoHead3: Csv.Sample = {
  data: {
    '0': '7',
    '1': '8'
  },
  label: {
    '2': '9'
  }
};


describe('Csv Dataset', () => {
  it('should make a dataset from csv', async () => {
    const dataset = await makeDatasetFromCsv({
      trainData: csvDataWithHead,
      testData: csvDataWithHead,
      validData: undefined,
      hasHeader: true,
      labels: [ 'C' ]
    });

    const metadata: Csv.DatasetMeta = {
      type: Types.DatasetType.Table,
      size: { train: 3, test: 3, valid: 0 },
      labelMap: undefined
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sample2, sample3 ]);
    expect(await dataset.train.nextBatch(1)).to.eql([]);
    expect(await dataset.test.nextBatch(1)).to.eql([ sample2 ]);
  });
  it('should make a dataset from csv with valid', async () => {
    const dataset = await makeDatasetFromCsv({
      trainData: csvDataWithHead,
      testData: csvDataWithHead,
      validData: csvDataWithHead,
      hasHeader: true,
      labels: [ 'C' ]
    });

    const metadata: Csv.DatasetMeta = {
      type: Types.DatasetType.Table,
      size: { train: 3, test: 3, valid: 3 },
      labelMap: undefined
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.valid.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sample2, sample3 ]);
    expect(await dataset.test.nextBatch(1)).to.eql([ sample2 ]);
    expect(await dataset.valid.nextBatch(1)).to.eql([ sample2 ]);
  });

  it('should make a dataset from csv without head', async () => {
    const dataset = await makeDatasetFromCsv({
      trainData: csvDataWithoutHead,
      testData: csvDataWithoutHead,
      validData: csvDataWithoutHead,
      hasHeader: false,
      labels: [ '2' ]
    });

    const metadata: Csv.DatasetMeta = {
      type: Types.DatasetType.Table,
      size: { train: 3, test: 3, valid: 3 },
      labelMap: undefined
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sampleNoHead1);
    expect(await dataset.test.next()).to.eql(sampleNoHead1);
    expect(await dataset.valid.next()).to.eql(sampleNoHead1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sampleNoHead2, sampleNoHead3 ]);
    expect(await dataset.test.nextBatch(1)).to.eql([ sampleNoHead2 ]);
    expect(await dataset.valid.nextBatch(1)).to.eql([ sampleNoHead2 ]);
  });
})
