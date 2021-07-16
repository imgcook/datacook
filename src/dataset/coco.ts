import { Dataset, Sample, DatasetType, Coco } from './types';
import { DatasetData, makeDataset } from './utils';

async function checkCocoMeta(metaObj: Record<string, any>) {
  if (!Array.isArray(metaObj.images)) {
    throw new TypeError('images should be array');
  }
  if (!Array.isArray(metaObj.annotations)) {
    throw new TypeError('annotations should be array');
  }
  metaObj.images.forEach((image) => {
    if (typeof image.id !== 'number') {
      throw new TypeError('invalid id field found in image data');
    }
    if (typeof image.width !== 'number') {
      throw new TypeError('invalid width field found in image data');
    }
    if (typeof image.height !== 'number') {
      throw new TypeError('invalid height field found in image data');
    }
    if (
      typeof image.url !== 'string'
      && typeof image.coco_url !== 'string'
      && typeof image.flickr_url !== 'string'
    ) {
      throw new TypeError('invalid url/flickr_url/coco_url field found in image data');
    }
  });
}

function cocoMetaToDatasetData(cocoMeta: Coco.Meta): Array<Sample<Coco.Image, Coco.Label>> {
  const annotationMap: Record<number, Array<Coco.Annotation>> = {};
  for (const ann of cocoMeta.annotations) {
    if (!annotationMap[ann.image_id]) {
      annotationMap[ann.image_id] = [];
    }
    annotationMap[ann.image_id].push(ann);
  }
  return cocoMeta.images.map((img: Coco.Image) => ({ data: img, label: annotationMap[img.id] }));
}

async function process(
  annotationObj: Coco.Meta
): Promise<{
  meta: Coco.Meta,
  datasetData: Array<Sample<Coco.Image, Coco.Label>>
}> {
  await checkCocoMeta(annotationObj);
  return { meta: annotationObj as Coco.Meta, datasetData: cocoMetaToDatasetData(annotationObj) };
}

export const makeDatasetFromCocoFormat = async (options: Coco.Options): Promise<Dataset<Sample<Coco.Image, Coco.Label>, Coco.DatasetMeta>> => {
  const { meta: trainMeta, datasetData: trainDatasetData } = await process(options.trainAnnotationObj);
  const { datasetData: testDatasetData } = await process(options.testAnnotationObj);
  let validDatasetData = undefined;
  if (options.validAnnotationObj) {
    validDatasetData = (await process(options.validAnnotationObj)).datasetData;
  }
  const data: DatasetData<Sample<Coco.Image, Coco.Label>> = {
    trainData: trainDatasetData,
    testData: testDatasetData,
    validData: validDatasetData
  };
  const labelMap: Record<number, Coco.Category> = {};
  (trainMeta as Coco.Meta).categories.forEach((category: Coco.Category) => {
    labelMap[category.id] = category;
  });
  const datasetMeta: Coco.DatasetMeta = {
    type: DatasetType.Image,
    size: {
      train: trainDatasetData.length,
      test: testDatasetData.length,
      valid: Array.isArray(validDatasetData) ? validDatasetData.length : 0
    },
    labelMap,
    info: trainMeta.info,
    licenses: trainMeta.licenses
  };
  return makeDataset(data, datasetMeta);
};
