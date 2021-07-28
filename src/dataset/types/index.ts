export type DefaultType = any;

// base sample
export interface BaseSample<T = DefaultType> {
  data: T;
}

export interface Sample<T = DefaultType, L = DefaultType> extends BaseSample<T> {
  label: L;
}

/**
 * table column type
 */
export enum TableColumnType {
  Number,
  String,
  Bool,
  Map,
  Datetime,
  Unknown
}

/**
 * structure of colume description
 */
export interface TableColumn {
  name: string,
  type: TableColumnType
}

/**
 * table schema for all columns
 */
export type TableSchema = Array<TableColumn>;

/**
 * data source type
 *   Table: data from db, csv
 *   Image: image data
 */
export enum DatasetType { Table, Image, Sound, General }

/**
 * size of data source
 */
export interface DatasetSize {
  train: number;
  test: number;
  valid?: number;
}

export interface ImageDimension {
  x: number,
  y: number,
  z: number
}

export interface BaseDatasetMeta {
  type: DatasetType;
  size?: DatasetSize;
}

export interface DatasetMeta extends BaseDatasetMeta {
  labelMap?: Record<number, any>;
}

/**
 * image data source metadata
 */
export interface ImageDatasetMeta extends DatasetMeta {
  dimension: ImageDimension;
}

/**
 * table data source metadata
 */
export interface TableDatasetMeta extends DatasetMeta {
  tableSchema: TableSchema;
  dataKeys: Array<string> | null;
}

export interface DataAccessor<T> {
  next: () => Promise<T | null>;
  nextBatch: (batchSize: number) => Promise<Array<T>>;
  seek: (pos: number) => Promise<void>;
  shuffle: (seed?: string) => void;
}

/**
 * data source api
 */
export interface Dataset<T extends Sample, D extends DatasetMeta> {
  // fetch data source metadata
  getDatasetMeta: () => Promise<D>;
  // test dataset accessor
  test: DataAccessor<T>;
  // train dataset accessor
  train: DataAccessor<T>;
  // validation dataset accessor, qoptional
  valid?: DataAccessor<T>;
  // suhffle interface
  shuffle: (seed?: string) => void;
}

export * as ObjectDetection from './object-detection';

export * as Coco from './coco';
export * as PascalVoc from './pascal-voc';
export * as Csv from './csv';
