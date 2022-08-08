export abstract class Vector {
  public data: number[];
  public length: number;
  constructor(data: number[]) {
    this.data = data;
    this.length = this.data.length;
  }
  abstract get(i: number): number;
}
