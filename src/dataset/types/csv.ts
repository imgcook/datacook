import { Sample as BaseSample, DatasetSize, DatasetType, BaseDatasetMeta } from './';

export interface Options {
  csvData: string;
}

export type ColumnData = number | string | boolean;

export type Label = Array<ColumnData>;

export type Sample = BaseSample<Record<string, ColumnData>, Label>;

export interface DatasetMeta extends BaseDatasetMeta {
  
}
