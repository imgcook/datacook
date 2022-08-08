import { Vector as VectorBase } from '../../core/classes';
export class Vector extends VectorBase{
  public data: number[];
  public length: number;
  constructor(data: number[]) {
    super(data);
    this.data = data;
    this.length = this.data.length;
  }
  public get(i: number): number {
    return this.data[i];
  }
}
