
import { DatasetType, BaseDatasetMeta, Sample as BaseSample } from '.';

export interface Source {
  database: string;
  annotation: string;
  image: string;
  flickrid: string;
}

export interface Owner {
  name: string;
  flickrid: string;
}

export interface Size {
  width: number;
  height: number;
  depth: number;
}

export interface Bndbox {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
}

export interface PascalVocObject {
  name: string;
  pose: string;
  truncated: 0 | 1;
  difficult: number;
  bndbox: Bndbox;
}

export interface ExtPascalVocObject extends PascalVocObject{
  id: number;
}

export interface Annotation {
  folder: string;
  filename: string;
  path: string;
  source: Source;
  owner: Owner;
  size: Size;
  segmented: number;
  object: Array<PascalVocObject>;
}

export interface ExtAnnotation {
  folder: string;
  filename: string;
  path: string;
  source: Source;
  owner: Owner;
  size: Size;
  segmented: number;
  object: Array<ExtPascalVocObject>;
}

export interface Options {
  trainAnnotationList: Array<Annotation>;
  testAnnotationList: Array<Annotation>;
  validAnnotationList?: Array<Annotation>;
}

export interface DatasetMeta extends BaseDatasetMeta {
  type: DatasetType.Image,
  labelMap: Array<string>;
}
export type Label = Array<ExtPascalVocObject>;
export type Sample = BaseSample<ExtAnnotation, Label>;
