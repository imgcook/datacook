import { Sample as BaseSample } from '.';

export type Label = string;

export interface ImageClassificationData {
  uri?: string;
  buffer?: ArrayBuffer;
}

export type Sample = BaseSample<ImageClassificationData, Label>;

export interface ImageDesc {
  category: string;
  uri?: string;
  buffer?: ArrayBuffer;
}

export type ImageList = Array<ImageDesc>;
