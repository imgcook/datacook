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
  public setColumn(i: number, x: Vector | number[]): void {
    let data: number[];
    if (x instanceof Vector) {
      data = x.data;
    } else {
      data = x;
    }
    if (data.length !== this.shape[0]) {
      throw new TypeError(`Vector length ${data.length} does not match, expect to be ${this.shape[0]}`);
    }
    for (let j = 0; j < this.shape[0]; j++) {
      this.data[j][i] = data[j];
    }
  }
  public set(i: number, j: number, x: number): void {
    this.data[i][j] = x;
  }
  public hahaha(): void {
    console.log('hhh');
  }
}
