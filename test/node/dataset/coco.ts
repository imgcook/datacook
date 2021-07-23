import { expect } from 'chai';
import { makeDataset, Types } from '../../../src/dataset';
import { makeDatasetFromCocoFormat } from '../../../src/dataset/format/coco';
import 'mocha';
import { Coco } from '../../../src/dataset/types';
import { cocoAnnotation as annotationObj } from './data';

const sample1: Types.Sample<Coco.Image, Coco.Label> = {
  data: annotationObj.images[0],
  label: [ annotationObj.annotations[0], annotationObj.annotations[1] ]
};

const sample2: Types.Sample<Coco.Image, Coco.Label> = {
  data: annotationObj.images[1],
  label: [ annotationObj.annotations[2], annotationObj.annotations[3] ]
};

const sample3: Types.Sample<Coco.Image, Coco.Label> = {
  data: annotationObj.images[2],
  label: [ annotationObj.annotations[4] ]
};

describe('Coco Dataset', () => {
  it('should make a dataset from coco', async () => {
    const dataset = await makeDatasetFromCocoFormat({
      trainAnnotationObj: annotationObj,
      testAnnotationObj: annotationObj
    });

    const metadata: Coco.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 0 },
      labelMap: {
        '1': { supercategory: 'abovePicture', id: 1, name: 'abovePicture' },
        '2': { supercategory: 'button', id: 2, name: 'button' }
      },
      info: undefined,
      licenses: undefined
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([sample2, sample3]);
    expect(await dataset.train.nextBatch(1)).to.eql([]);
    expect(await dataset.test.nextBatch(1)).to.eql([sample2]);
  });
  it('should make a dataset from coco with valid', async () => {
    const dataset = await makeDatasetFromCocoFormat({
      trainAnnotationObj: annotationObj,
      testAnnotationObj: annotationObj,
      validAnnotationObj: annotationObj
    });

    const metadata: Coco.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 3 },
      labelMap: {
        '1': { supercategory: 'abovePicture', id: 1, name: 'abovePicture' },
        '2': { supercategory: 'button', id: 2, name: 'button' }
      },
      info: undefined,
      licenses: undefined
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.valid.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([sample2, sample3]);
    expect(await dataset.test.nextBatch(1)).to.eql([sample2]);
    expect(await dataset.valid.nextBatch(1)).to.eql([sample2]);
  });
})
