import { Sample as BaseSample } from './';

export type Bbox = [
  x: number,
  y: number,
  width: number,
  height: number
];

export type Label = Array<{ id: number; bbox: Bbox; }>;

export interface ObjectDetectionData {
  uri?: string;
  buffer?: ArrayBuffer;
}

export type Sample = BaseSample<ObjectDetectionData, Label>;

