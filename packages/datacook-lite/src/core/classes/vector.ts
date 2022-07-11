export abstract class Vector {
  public data: number[];
  public length: number;
  public grad: Vector;
  constructor(data: number[]) {
    this.data = data;
    this.length = this.data.length;
  }
  abstract get(i: number): number;
  abstract set(i: number): void;
  abstract backward(grad: Vector): void;
}
