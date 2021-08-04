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

export interface ImageDimension {
  x: number,
  y: number,
  z: number
}

export interface Dataset<T> {
  next: () => Promise<T | null>;
  nextBatch: (batchSize: number) => Promise<Array<T>>;
  seek: (pos: number) => Promise<void>;
  shuffle: (seed?: string) => void;
}

export * as ObjectDetection from './object-detection';
export * as ImageClassification from './image-classification';

export * as Coco from './coco';
export * as PascalVoc from './pascal-voc';
export * as Csv from './csv';
