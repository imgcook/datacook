import { Vector as VectorBase } from '../../core/classes';
import { Denpendency } from "../../core/classes";
import { add1d } from '../op';
import { createOneVector, createZeroVector } from './creation';
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
  public backward(grad?: VectorBase): void {
    if (!this.grad) {
      if (grad) {
        this.grad = grad;
      } else {
        grad = createOneVector(this.length);
        this.grad = grad;
      }
    } else {
      if (grad)
        this.grad = add1d(this.grad, grad);
    }
    this.dependency.forEach((dep: Denpendency) => {
      const targetGrad = dep.gradFunc(this.grad);
      dep.target.backward(targetGrad);
    });
  }
  public zeroGrad(): void {
    this.grad = createZeroVector(this.length);
  }
  public assign(val: number[]): void {
    for (let i = 0; i < this.length; i++) {
      this.data[i] = val[i];
    }
  }
}
