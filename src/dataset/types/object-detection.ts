import { Sample as BaseSample, Dataset as BaseDataset, DatasetMeta as BaseDatasetMeta } from './';

export type Bbox = [
  x: number,
  y: number,
  width: number,
  height: number
];

export type Label = Array<{ id: number; bbox: Bbox; }>;

export type Sample = BaseSample<string, Label>;

export interface DatasetMeta extends BaseDatasetMeta {
  labelMap: Record<number, string>;
}

export type Dataset = BaseDataset<Sample, DatasetMeta>;
