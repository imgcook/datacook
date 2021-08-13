import { Dataset, PascalVoc } from '../types';
import { ArrayDatasetImpl } from '../utils';

function isAnnotationArray(arg: PascalVoc.PascalVocObject[] | PascalVoc.PascalVocObject): arg is Array<PascalVoc.PascalVocObject> {
  return Array.isArray(arg);
}

export function makeDatasetFromPascalVoc(annotations: Array<PascalVoc.Annotation>): Dataset<PascalVoc.Sample> {
  const samples: PascalVoc.Sample[] = annotations.map((annotation: PascalVoc.Annotation) => {
    return {
      data: annotation,
      label: isAnnotationArray(annotation.annotation.object) ? annotation.annotation.object : [ annotation.annotation.object ]
    };
  });
  return new ArrayDatasetImpl<PascalVoc.Sample>(samples);
}

export function extractCategoriesFromPascalVoc(annotations: Array<PascalVoc.Annotation>): Array<string> {
  const labelSet = new Set<string>();
  annotations.forEach((annotation: PascalVoc.Annotation) => {
    if (isAnnotationArray(annotation.annotation.object)) {
      annotation.annotation.object.forEach((obj) => {
        labelSet.add(obj.name);
      })
    } else {
      labelSet.add(annotation.annotation.object.name);
    }
  });
  return Array.from(labelSet);
}
