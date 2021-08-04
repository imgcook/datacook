import { Sample as BaseSample } from './';
import { Coco } from './';

export type Bbox = Coco.Bbox;

export type Label = Array<{ name: string; bbox: Bbox; }>;

export interface ObjectDetectionData {
  uri?: string;
  buffer?: ArrayBuffer;
}

export type Sample = BaseSample<ObjectDetectionData, Label>;
