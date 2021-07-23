import * as papaparse from 'papaparse';

import { Dataset, Sample, DatasetType, Csv } from '../types';
import { DatasetData, makeDataset } from '../utils';

export const makeDatasetFromCsv = async (options: Csv.Options): Promise<Dataset<Csv.Sample, Csv.DatasetMeta>> => {
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
