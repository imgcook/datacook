
import { Sample as BaseSample } from '.';

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

export type Label = Array<PascalVocObject>;
export type Sample = BaseSample<Annotation, Label>;
