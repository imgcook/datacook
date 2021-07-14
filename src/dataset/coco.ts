import { Dataset, Sample, DatasetType } from './types';
import { DatasetData, makeDataset } from './utils';
import * as fs from 'fs-extra';
import * as path from 'path';
import { train } from '@tensorflow/tfjs-core';

export type Image = {
  id: number;
  width: number;
  height: number;
  file_name: string;
  license?: number;
  flickr_url?: string;
  coco_url?: string;
  url?: string;
  date_captured?: string;
};

export type Bbox = Array<[x:number, y:number, width: number, height: number]>;

export type Annotation = {
  id: number;
  image_id: number;
  category_id: number;
  segmentation?: Array<any>;
  area: number;
  bbox: Bbox;
  iscrowd: 0 | 1;
};

export type Category = {
  supercategory: string;
	id: number;
	name: string;
};

export type CocoMeta = {
  info: any;
  licenses: Array<any>;
  images: Array<Image>;
  annotations: Array<Annotation>;
  categories: Array<Category>;
};

export type CocoOptions = {
  trainDir: string;
  trainAnnotationFile?: string;
  testDir: string;
  testAnnotationFile?: string;
  validDir?: string;
  validAnnotationFile?: string;
};

async function checkCocoMeta(metaObj: Record<string, any>) {
  
}

export type Label = Array<{bbox: Bbox, categoryId: number}>;

function cocoMetaToDatasetData(cocoMeta: CocoMeta): Array<Sample<string, Label>> {
  const annotationMap: Record<number, Record<number, Bbox>> = {};
  for (const ann of cocoMeta.annotations) {
    if (!annotationMap[ann.image_id]) {
      annotationMap[ann.image_id] = {};
    }
    if (!annotationMap[ann.image_id][ann.category_id]) {
      annotationMap[ann.image_id][ann.category_id] = [];
    }
    annotationMap[ann.image_id][ann.category_id].concat(ann.bbox);
  };
  return cocoMeta.images.map((img: Image) => {
    const sample: Sample<string, Label> = {
      data: img.coco_url || img.flickr_url || img.url,
      label: []
    };
    const categories = annotationMap[img.id];
    if (!categories) {
      throw new TypeError(`no annotation found for image: ${img.file_name || img.coco_url || img.flickr_url || img.url}`);
    }
    for (const categoryId in categories) {
      sample.label.push({
        categoryId: Number(categoryId),
        bbox: categories[categoryId]
      });
    }
    return sample;
  });
}



export const makeDatasetFromCocoFormat = async (options: CocoOptions): Promise<Dataset<Sample<string, Label>, any>> => {
  const trainAnnFile = options.trainAnnotationFile ?
    options.trainAnnotationFile : path.join(options.trainDir, 'annotation.json');
  const trainMeta = await fs.readJson(trainAnnFile);
  await checkCocoMeta(trainMeta);
  const testAnnFile = options.testAnnotationFile ?
    options.testAnnotationFile : path.join(options.testDir, 'annotation.json');
  const testMeta = await fs.readJson(testAnnFile);
  await checkCocoMeta(testMeta);

  const trainData = cocoMetaToDatasetData(trainMeta);
  const testData = cocoMetaToDatasetData(testMeta);
  let validData = undefined;
  if (options.validDir) {
    const validAnnFile = options.validAnnotationFile ?
      options.validAnnotationFile : path.join(options.validDir, 'annotation.json');
    const validMeta = await fs.readJson(validAnnFile);
    await checkCocoMeta(validMeta);
    validData = cocoMetaToDatasetData(trainMeta);
  }
  const data: DatasetData<Sample<string, Label>> = {
    trainData,
    testData,
    validData
  };
  const labelMap: Record<number, Category> = {};
  (trainMeta as CocoMeta).categories.forEach((category) => {
    labelMap[category.id] = category;
  });
  const datasetMeta = {
    type: DatasetType.Image,
    size: {
      train: trainData.length,
      test: testData.length,
      valid: validData?.length
    },
    labelMap
  };
  makeDataset(data, datasetMeta);

  return null;
};
