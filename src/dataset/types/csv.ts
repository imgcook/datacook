import { Sample as BaseSample } from './';

export type ColumnData = string;

export type Label = Record<string, ColumnData>;

export type Sample = BaseSample<Record<string, ColumnData>, Label>;
