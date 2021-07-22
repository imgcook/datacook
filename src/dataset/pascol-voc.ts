import { Dataset, Sample, DatasetType, PascolVoc } from './types';
import { makeDataset } from './utils';

function attachId(labelMap: Array<string>, annotationList: Array<PascolVoc.Annotation>): Array<Sample<PascolVoc.ExtAnnotation, Array<PascolVoc.ExtPascolVocObject>>> {
  return annotationList.map((annotation: PascolVoc.Annotation) => {
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

export const makeDatasetFromPascolVocFormat = async (options: PascolVoc.Options): Promise<Dataset<Sample<PascolVoc.ExtAnnotation, Array<PascolVoc.ExtPascolVocObject>>, PascolVoc.DatasetMeta>> => {
  const labelNames: Array<string> = [];
  const trainData = options.trainAnnotationList.map((annotation: PascolVoc.Annotation) => {
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
  const validData = attachId(labelNames, options.validAnnotationList);

  const datasetMeta: PascolVoc.DatasetMeta = {
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
