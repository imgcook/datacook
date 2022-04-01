export class Vector {
  public data: number[];
  public length: number;
  constructor(data: number[]) {
    this.data = data;
    this.length = this.data.length;
  }
  public get(i: number): number {
    return this.data[i];
  }
}
