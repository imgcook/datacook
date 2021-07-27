import { expect } from 'chai';
import { Types } from '../../../src/dataset';
import { makeObjectDetectionDatasetFromCoco, makeObjectDetectionDatasetFromPascalVoc } from '../../../src/dataset/object-detection';
import 'mocha';
import { ObjectDetection } from '../../../src/dataset/types';
import { cocoAnnotation, pascalVocAnnotation } from './data';
import * as path from 'path';

describe('Objec detection dataset', () => {
  it('should make a dataset from coco', async () => {
    const sample1: Types.ObjectDetection.Sample = {
      data: { uri: cocoAnnotation.images[0].url },
      label: [{
        id: cocoAnnotation.annotations[0].id,
        bbox: cocoAnnotation.annotations[0].bbox
      }, {
        id: cocoAnnotation.annotations[1].id,
        bbox: cocoAnnotation.annotations[1].bbox
      }]
    };
    
    const sample2: Types.ObjectDetection.Sample = {
      data: { uri: cocoAnnotation.images[1].url },
      label: [{
        id: cocoAnnotation.annotations[2].id,
        bbox: cocoAnnotation.annotations[2].bbox
      }, {
        id: cocoAnnotation.annotations[3].id,
        bbox: cocoAnnotation.annotations[3].bbox
      }]
    };
    
    const sample3: Types.ObjectDetection.Sample = {
      data: { uri: cocoAnnotation.images[2].url },
      label: [{
        id: cocoAnnotation.annotations[4].id,
        bbox: cocoAnnotation.annotations[4].bbox
      }]
    };
    
    const dataset = await makeObjectDetectionDatasetFromCoco({
      trainAnnotationObj: cocoAnnotation,
      testAnnotationObj: cocoAnnotation
    });

    const metadata: ObjectDetection.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 0 },
      labelMap: {
        '1': 'abovePicture',
        '2': 'button'
      }
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([sample2, sample3]);
    expect(await dataset.train.nextBatch(1)).to.eql([]);
    expect(await dataset.test.nextBatch(1)).to.eql([sample2]);
  });

  it('should make a dataset from pascol', async () => {
    const sample1: Types.ObjectDetection.Sample = {
      data: { uri: pascalVocAnnotation[0].path },
      label: [ {
        id: 0,
        bbox: [
          pascalVocAnnotation[0].object[0].bndbox.xmin,
          pascalVocAnnotation[0].object[0].bndbox.ymin,
          pascalVocAnnotation[0].object[0].bndbox.xmax - pascalVocAnnotation[0].object[0].bndbox.xmin,
          pascalVocAnnotation[0].object[0].bndbox.ymax - pascalVocAnnotation[0].object[0].bndbox.ymin
        ],
        }, {
          id: 1,
          bbox: [
            pascalVocAnnotation[0].object[1].bndbox.xmin,
            pascalVocAnnotation[0].object[1].bndbox.ymin,
            pascalVocAnnotation[0].object[1].bndbox.xmax - pascalVocAnnotation[0].object[1].bndbox.xmin,
            pascalVocAnnotation[0].object[1].bndbox.ymax - pascalVocAnnotation[0].object[1].bndbox.ymin
          ]
        }
      ]
    };
    
    const sample2: Types.ObjectDetection.Sample = {
      data: { uri: pascalVocAnnotation[1].path },
      label: [ {
        id: 0,
        bbox: [
          pascalVocAnnotation[1].object[0].bndbox.xmin,
          pascalVocAnnotation[1].object[0].bndbox.ymin,
          pascalVocAnnotation[1].object[0].bndbox.xmax - pascalVocAnnotation[1].object[0].bndbox.xmin,
          pascalVocAnnotation[1].object[0].bndbox.ymax - pascalVocAnnotation[1].object[0].bndbox.ymin
        ],
        }, {
          id: 1,
          bbox: [
            pascalVocAnnotation[1].object[1].bndbox.xmin,
            pascalVocAnnotation[1].object[1].bndbox.ymin,
            pascalVocAnnotation[1].object[1].bndbox.xmax - pascalVocAnnotation[1].object[1].bndbox.xmin,
            pascalVocAnnotation[1].object[1].bndbox.ymax - pascalVocAnnotation[1].object[1].bndbox.ymin
          ]
        }
      ]
    };
    
    const sample3: Types.ObjectDetection.Sample = {
      data: { uri: pascalVocAnnotation[2].path },
      label: [
        {
          id: 2,
          bbox: [
            pascalVocAnnotation[2].object[0].bndbox.xmin,
            pascalVocAnnotation[2].object[0].bndbox.ymin,
            pascalVocAnnotation[2].object[0].bndbox.xmax - pascalVocAnnotation[2].object[0].bndbox.xmin,
            pascalVocAnnotation[2].object[0].bndbox.ymax - pascalVocAnnotation[2].object[0].bndbox.ymin
          ],
        },
        {
          id: 3,
          bbox: [
            pascalVocAnnotation[2].object[1].bndbox.xmin,
            pascalVocAnnotation[2].object[1].bndbox.ymin,
            pascalVocAnnotation[2].object[1].bndbox.xmax - pascalVocAnnotation[2].object[1].bndbox.xmin,
            pascalVocAnnotation[2].object[1].bndbox.ymax - pascalVocAnnotation[2].object[1].bndbox.ymin
          ]
        }
      ]
    };
    
    const dataset = await makeObjectDetectionDatasetFromPascalVoc({
      trainAnnotationList: pascalVocAnnotation,
      testAnnotationList: pascalVocAnnotation
    });

    const metadata: Types.ObjectDetection.DatasetMeta = {
      type: Types.DatasetType.Image,
      size: { train: 3, test: 3, valid: 0 },
      labelMap: {
        '0': 'dog',
        '1': 'person',
        '2': 'dog2',
        '3': 'person2'
      }
    };

    expect(await dataset.getDatasetMeta()).to.eql(metadata);
    expect(await dataset.train.next()).to.eql(sample1);
    expect(await dataset.test.next()).to.eql(sample1);
    expect(await dataset.train.nextBatch(2)).to.eql([ sample2, sample3 ]);
    expect(await dataset.train.nextBatch(1)).to.eql([]);
    expect(await dataset.test.nextBatch(1)).to.eql([sample2]);
  });
})
