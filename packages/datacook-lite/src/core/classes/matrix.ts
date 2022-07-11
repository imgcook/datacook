import { Vector } from "./vector";

interface Denpendency {
  gradFunc: (data: Matrix) => Matrix | Vector | number;
  dep: Matrix | Vector | number;
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
  abstract backward(grad?: Matrix): void;
}


