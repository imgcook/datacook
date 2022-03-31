import { Sample as BaseSample } from '.';

export type Label = string;

export type TextClassificationData = string;

export type Sample = BaseSample<TextClassificationData, Label>;

export interface TextDesc {
  category: string;
  text: string;
}

export type TextList = Array<TextDesc>;
