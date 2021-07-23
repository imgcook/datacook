import { Sample as BaseSample, DatasetMeta as BaseDatasetMeta } from './';

export interface Options {
  trainData: string;
  testData: string;
  validData?: string;
  hasHeader: boolean;
  delimiter?: string;
  labels?: string[];
}

export type ColumnData = string;

export type Label = Record<string, ColumnData>;

export type Sample = BaseSample<Record<string, ColumnData>, Label>;

export interface DatasetMeta extends BaseDatasetMeta {
  labelMap: undefined;
}
