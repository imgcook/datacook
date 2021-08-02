import { Dataset, PascalVoc } from '../types';
import { ArrayDatasetImpl } from '../utils';

export function makeDatasetFromPascalVoc(annotations: Array<PascalVoc.Annotation>): Dataset<PascalVoc.Sample> {
  const samples = annotations.map((annotation: PascalVoc.Annotation) => {
    return {
      data: annotation,
      label: annotation.object
    };
  });
  return new ArrayDatasetImpl<PascalVoc.Sample>(samples);
}

export function extractCategoriesFromPascalVoc(annotations: Array<PascalVoc.Annotation>): Array<string> {
  const labelSet = new Set<string>();
  annotations.forEach((annotation: PascalVoc.Annotation) => {
    annotation.object?.forEach((obj) => {
      labelSet.add(obj.name);
    });
  });
  return Array.from(labelSet);
}
