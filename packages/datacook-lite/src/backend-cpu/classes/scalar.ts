import { Scalar as ScalarBase } from '../../core/classes';
import { scalar } from '../../core/classes/creation';
import { Denpendency } from "../../core/classes";
export class Scalar extends ScalarBase {
  public data: number;
  constructor(data: number) {
    super(data);
    this.data = data;
  }
  public values(): number {
    return this.data;
  }
  public backward(grad?: ScalarBase): void {
    if (!this.grad) {
      if (grad) {
        this.grad = grad;
      } else {
        grad = scalar(1);
        this.grad = grad;
      }
    } else {
      if (grad)
        this.grad = scalar(this.grad.data + grad.data);
    }
    this.dependency.forEach((dep: Denpendency) => {
      const targetGrad = dep.gradFunc(grad);
      dep.target.backward(targetGrad);
    });
  }
}
