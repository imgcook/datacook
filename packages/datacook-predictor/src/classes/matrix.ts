import { Vector } from "./vector";

export class Matrix {
  public data: number[][];
  public shape: number[];
  public constructor(data: number[][]) {
    this.data = data;
    this.shape = [ data.length, data[0].length ];
  }
  public getColumn(i: number): Vector {
    return new Vector(this.data.map((d) => d[i]));
  }
  public getRow(i: number): Vector {
    return new Vector(this.data[i]);
  }
  public get(i: number, j: number): number {
    return this.data[i][j];
  }

}
