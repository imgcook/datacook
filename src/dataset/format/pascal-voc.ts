import { Dataset, DatasetType, PascalVoc } from '../types';
import { makeDataset } from '../utils';

function attachId(labelMap: Array<string>, annotationList: Array<PascalVoc.Annotation>): Array<PascalVoc.Sample> {
  return annotationList.map((annotation: PascalVoc.Annotation) => {
    const extObjs = annotation.object?.map((obj) => {
      const index = labelMap.indexOf(obj.name);
      if (index >= 0) {
        return {
          ...obj,
          id: index
        };
      } else {
        throw TypeError(`'${obj.name}' not exists in train dataset.`);
      }
    });
    return {
      data: {
        ...annotation,
        object: extObjs
      },
      label: extObjs
    };
  });
}

export const makeDatasetFromPascalVocFormat = async (options: PascalVoc.Options): Promise<Dataset<PascalVoc.Sample, PascalVoc.DatasetMeta>> => {
  const labelNames: Array<string> = [];
  const trainData = options.trainAnnotationList.map((annotation: PascalVoc.Annotation) => {
    const extObjs = annotation.object?.map((obj) => {
      const index = labelNames.indexOf(obj.name);
      if (index >= 0) {
        return {
          ...obj,
          id: index
        };
      } else {
        labelNames.push(obj.name);
        return {
          ...obj,
          id: labelNames.length - 1
        };
      }
    });
    return {
      data: {
        ...annotation,
        object: extObjs
      },
      label: extObjs
    };
  });
  const testData = attachId(labelNames, options.testAnnotationList);
  const validData = Array.isArray(options.validAnnotationList) ? attachId(labelNames, options.validAnnotationList) : undefined;

  const datasetMeta: PascalVoc.DatasetMeta = {
    type: DatasetType.Image,
    size: {
      train: trainData.length,
      test: testData.length,
      valid: Array.isArray(validData) ? validData.length : 0
    },
    labelMap: labelNames
  };
  return makeDataset(
    {
      trainData: trainData,
      testData: testData,
      validData: validData
    },
    datasetMeta
  );
};
