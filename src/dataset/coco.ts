import { Dataset, Sample, DatasetType, Coco } from './types';
import { DatasetData, makeDataset } from './utils';
import * as fs from 'fs-extra';
import * as path from 'path';
import * as assert from 'assert';

async function checkCocoMeta(metaObj: Record<string, any>) {
  assert.ok(Array.isArray(metaObj.images), 'images should be array');
  assert.ok(Array.isArray(metaObj.annotations), 'annotations should be array');
  metaObj.images.forEach(image => {
    assert.ok(typeof image.id === 'number', 'invalid id field found in image data');
    assert.ok(typeof image.width === 'number', 'invalid width field found in image data');
    assert.ok(typeof image.height === 'number', 'invalid height field found in image data');
    assert.ok(
      typeof image.url === 'string'
      || typeof image.coco_url === 'string'
      || typeof image.flickr_url === 'string',
      'invalid url/flickr_url/coco_url field found in image data'
    );
  });
}

function cocoMetaToDatasetData(cocoMeta: Coco.Meta): Array<Sample<Coco.Image, Coco.Label>> {
  const annotationMap: Record<number, Array<Coco.Annotation>> = {};
  for (const ann of cocoMeta.annotations) {
    if (!annotationMap[ann.image_id]) {
      annotationMap[ann.image_id] = [];
    }
    annotationMap[ann.image_id].push(ann);
  };
  return cocoMeta.images.map((img: Coco.Image) => ({ data: img, label: annotationMap[img.id]}));
}

async function process(
  dir: string,
  annotationFile?: string
): Promise<{
  meta: Coco.Meta,
  datasetData: Array<Sample<Coco.Image, Coco.Label>>
}> {
  const trainAnnFile = annotationFile ? annotationFile : path.join(dir, 'annotation.json');
  const meta = await fs.readJson(trainAnnFile);
  await checkCocoMeta(meta);
  return { meta, datasetData: cocoMetaToDatasetData(meta) };
}

export const makeDatasetFromCocoFormat = async (options: Coco.Options): Promise<Dataset<Sample<Coco.Image, Coco.Label>, Coco.DatasetMeta>> => {
  const { meta: trainMeta, datasetData: trainDatasetData } = await process(options.trainDir, options.trainAnnotationFile);
  const { datasetData: testDatasetData } = await process(options.testDir, options.testAnnotationFile);
  let validDatasetData = undefined;
  if (options.validDir) {
    validDatasetData = (await process(options.validDir, options.validAnnotationFile)).datasetData;
  }
  const data: DatasetData<Sample<Coco.Image, Coco.Label>> = {
    trainData: trainDatasetData,
    testData: testDatasetData,
    validData: validDatasetData
  };
  const labelMap: Record<number, Coco.Category> = {};
  (trainMeta as Coco.Meta).categories.forEach((category) => {
    labelMap[category.id] = category;
  });
  const datasetMeta: Coco.DatasetMeta = {
    type: DatasetType.Image,
    size: {
      train: trainDatasetData.length,
      test: testDatasetData.length,
      valid: validDatasetData?.length
    },
    labelMap,
    info: trainMeta.info,
    licenses: trainMeta.licenses
  };
  return makeDataset(data, datasetMeta);
};
