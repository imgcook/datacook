import { expect } from 'chai';
import { Types } from '../../../src/dataset';
import { makeDatasetFromPascalVocFormat } from '../../../src/dataset/format/pascal-voc';
import 'mocha';
import { PascalVoc } from '../../../src/dataset/types';
import * as path from 'path';
import { pascalVocAnnotation } from './data';

const sample1: Types.PascalVoc.Sample = {
  data: { ...pascalVocAnnotation[0], object: [
    { ...pascalVocAnnotation[0].object[0], id: 0 }, { ...pascalVocAnnotation[0].object[1], id: 1 }
  ]},
  label: [ { ...pascalVocAnnotation[0].object[0], id: 0 }, { ...pascalVocAnnotation[0].object[1], id: 1 }
  ]
};

const sample2: Types.PascalVoc.Sample = {
  data: { ...pascalVocAnnotation[1], object: [
    { ...pascalVocAnnotation[1].object[0], id: 0 }, { ...pascalVocAnnotation[1].object[1], id: 1 }
  ]},
  label: [ { ...pascalVocAnnotation[1].object[0], id: 0 }, { ...pascalVocAnnotation[1].object[1], id: 1 } ]
};

const sample3: Types.PascalVoc.Sample = {
  data: { ...pascalVocAnnotation[2], object: [
    { ...pascalVocAnnotation[2].object[0], id: 2 }, { ...pascalVocAnnotation[2].object[1], id: 3 }
  ]},
  label: [ { ...pascalVocAnnotation[2].object[0], id: 2 }, { ...pascalVocAnnotation[2].object[1], id: 3 } ]
};

describe('pascal-voc Dataset', () => {
  it('should make a dataset from pascalvoc', async () => {
    const dataset = await makeDatasetFromPascalVocFormat({
      trainAnnotationList: pascalVocAnnotation,
      testAnnotationList: pascalVocAnnotation
    });

    const metadata: Types.PascalVoc.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 0 },
      labelMap: [ 'dog', 'person', 'dog2', 'person2' ]
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sample2, sample3 ]);
    expect(await dataset.train.nextBatch(1)).to.eql([]);
    expect(await dataset.test.nextBatch(1)).to.eql([ sample2 ]);
  });
  it('should make a dataset from pascalvoc with valid', async () => {
    const dataset = await makeDatasetFromPascalVocFormat({
      trainAnnotationList: pascalVocAnnotation,
      testAnnotationList: pascalVocAnnotation,
      validAnnotationList: pascalVocAnnotation
    });

    const metadata: PascalVoc.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 3 },
      labelMap: [ 'dog', 'person', 'dog2', 'person2' ]
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.valid.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sample2, sample3 ]);
    expect(await dataset.test.nextBatch(1)).to.eql([ sample2 ]);
    expect(await dataset.valid.nextBatch(1)).to.eql([ sample2 ]);
  });
})
