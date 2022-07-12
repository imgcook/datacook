import { Vector as VectorBase } from '../../core/classes';
import { Denpendency } from "../../core/classes";
import { add1d } from '../op';
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
  public set(i: number, val: number): void {
    this.data[i] = val;
  }
  public values(): number[] {
    return this.data;
  }
  public backward(grad: VectorBase): void {
    this.grad = add1d(this.grad, grad);
    this.dependency.forEach((dep: Denpendency) => {
      const targetGrad = dep.gradFunc(grad);
      dep.target.backward(targetGrad);
    });
  }
}
