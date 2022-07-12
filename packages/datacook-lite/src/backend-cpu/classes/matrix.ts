import { Vector } from "./vector";
import { matrix, Matrix as MatrixBase } from '../../core/classes';
import { add2d } from "../op";
import { Denpendency } from "../../core/classes";
import { createOneMatrix } from "./creation";

export class Matrix extends MatrixBase {
  public data: number[][];
  public shape: number[];
  public constructor(data: number[][]) {
    super(data);
    this.data = data;
    this.shape = [ data.length, data[0].length ];
  }
  public getColumn(i: number): Vector {
    return new Vector(this.data.map((d) => d[i]));
  }
  public values(): number[][] {
    return this.data;
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
  public backward(grad?: MatrixBase): void {
    if (!this.grad) {
      if (grad) {
        this.grad = grad;
      } else {
        grad = createOneMatrix(this.shape[0], this.shape[1]);
        this.grad = grad;
      }
    } else {
      this.grad = add2d(this.grad, grad);
    }
    this.dependency.forEach((dep: Denpendency) => {
      const targetGrad = dep.gradFunc(grad);
      dep.target.backward(targetGrad);
    });
  }
}
