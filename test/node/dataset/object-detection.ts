import { expect } from 'chai';
import { Types } from '../../../src/dataset';
import { makeObjectDetectionDatasetFromCoco } from '../../../src/dataset/object-detection';
import 'mocha';
import { Coco, ObjectDetection } from '../../../src/dataset/types';

const annotationObj: Coco.Meta = {
  images: [
    {
      file_name: 'f984d880-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/f984d880-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 1,
      height: 286
    },
    {
      file_name: 'fb6a8870-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/fb6a8870-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 2,
      height: 363
    },
    {
      file_name: 'fd5abfb0-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/fd5abfb0-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 3,
      height: 286
    }
  ],
  annotations: [
    {
      image_id: 1,
      id: 1,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        36,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 1,
      id: 2,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        270,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 2,
      id: 3,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        270,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 2,
      id: 4,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        170,
        36,
        110,
        150
      ],
      category_id: 2
    },
    {
      image_id: 3,
      id: 5,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        150,
        136,
        110,
        50
      ],
      category_id: 1
    }
  ],
  categories: [
    {
      supercategory: 'abovePicture',
      id: 1,
      name: 'abovePicture'
    },
    {
      supercategory: 'button',
      id: 2,
      name: 'button'
    }
  ]
};

const sample1: Types.ObjectDetection.Sample = {
  data: annotationObj.images[0].url,
  label: [{
    id: annotationObj.annotations[0].id,
    bbox: annotationObj.annotations[0].bbox
  }, {
    id: annotationObj.annotations[1].id,
    bbox: annotationObj.annotations[1].bbox
  }]
};

const sample2: Types.ObjectDetection.Sample = {
  data: annotationObj.images[1].url,
  label: [{
    id: annotationObj.annotations[2].id,
    bbox: annotationObj.annotations[2].bbox
  }, {
    id: annotationObj.annotations[3].id,
    bbox: annotationObj.annotations[3].bbox
  }]
};

const sample3: Types.ObjectDetection.Sample = {
  data: annotationObj.images[2].url,
  label: [{
    id: annotationObj.annotations[4].id,
    bbox: annotationObj.annotations[4].bbox
  }]
};

describe('Coco Dataset', () => {
  it('should make a dataset from coco', async () => {
    const dataset = await makeObjectDetectionDatasetFromCoco({
      trainAnnotationObj: annotationObj,
      testAnnotationObj: annotationObj
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
})
