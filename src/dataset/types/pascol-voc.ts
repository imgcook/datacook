
import { BaseDatasetMeta } from './index';

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

export interface PascolVocObject {
  name: string;
  pose: string;
  truncated: 0 | 1;
  difficult: number;
  bndbox: Bndbox;
}

export interface ExtPascolVocObject extends PascolVocObject{
  id: number;
}

export interface Annotation {
  folder: string;
  filename: string;
  source: Source;
  owner: Owner;
  size: Size;
  segmented: number;
  object: Array<PascolVocObject>;
}

export interface ExtAnnotation {
  folder: string;
  filename: string;
  source: Source;
  owner: Owner;
  size: Size;
  segmented: number;
  object: Array<ExtPascolVocObject>;
}

export interface Options {
  trainAnnotationList: Array<Annotation>;
  testAnnotationList: Array<Annotation>;
  validAnnotationList: Array<Annotation>;
}

export interface DatasetMeta extends BaseDatasetMeta {
  labelMap: Array<string>;
}
