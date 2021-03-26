export type DefaultType = any;

// sample
export interface Sample<T = DefaultType> {
  label: number;
  data: T;
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
export enum DatasetType { Table, Image }

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

export interface DatasetMeta {
  type: DatasetType;
  size: DatasetSize;
  labelMap: Record<number, string>;
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
}

/**
 * data source api
 */
export interface Dataset<T = DefaultType, D = DatasetMeta> {
  // fetch data source metadata
  getDatasetMeta: () => Promise<D>;
  // test dataset accessor
  test: DataAccessor<T>;
  // train dataset accessor
  train: DataAccessor<T>;
  // validation dataset accessor, qoptional
  valid?: DataAccessor<T>;
}
