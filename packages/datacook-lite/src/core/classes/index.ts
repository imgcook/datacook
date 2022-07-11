export interface Denpendency {
  gradFunc: (data: Matrix) => Matrix | Vector | number;
  target: Matrix | Vector | number;
}

export abstract class Matrix {
  public data: number[][];
  public shape: number[];
  public grad: Matrix;
  public dependency: Array<Denpendency>;
  public constructor(data: number[][]) {
    this.shape = [ data.length, data[0].length ];
  }
  abstract getColumn(i: number): Vector;
  abstract getRow(i: number): Vector;
  abstract get(i: number, j: number): number;
  abstract setColumn(i: number, x: Vector | number[]): void;
  abstract set(i: number, j: number, x: number): void;
  abstract backward(grad: Matrix): void;
}


export abstract class Vector {
  public data: number[];
  public length: number;
  public grad: Vector;
  public dependency: Array<Denpendency>;
  constructor(data: number[]) {
    this.data = data;
    this.length = this.data.length;
  }
  abstract get(i: number): number;
  abstract set(i: number): void;
  abstract backward(grad: Vector): void;
}


export { createZeroMatrix, matrix, vector } from './creation';
