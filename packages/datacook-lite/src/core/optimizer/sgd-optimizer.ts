import { Vector } from "../../backend-cpu/classes";
import { mul1dForward, mul2dForward, sub2dForward, sub1dForward } from "../../backend-cpu/op/binary-op";
import { Matrix, Scalar } from "../classes";
import { Tensor } from "../classes";
import { Optimizer } from "./classes";

export interface SGDOptimizerProps {
  learningRate: number,
}

export class SGDOptimizer extends Optimizer {
  protected learningRate: number;
  constructor(params: Tensor[], props: SGDOptimizerProps) {
    super();
    if (!params) {
      throw new TypeError('Params should be provided');
    }
    if (!props) {
      throw new TypeError('Learning rate should be provided');
    }
    const { learningRate } = props;
    this.params = params;
    this.learningRate = learningRate;
  }
  step(): void {
    this.params.forEach((p: Tensor) => {
      if (p instanceof Matrix) {
        const val = sub2dForward(p, mul2dForward(p.grad, this.learningRate));
        p.assign(val.values());
      }
      if (p instanceof Vector) {
        const val = sub1dForward(p, mul1dForward(p.grad, this.learningRate));
        p.assign(val.values());
      }
      if (p instanceof Scalar) {
        const val = p.data - p.grad.data * this.learningRate;
        p.assign(val);
      }
    });
  }
}
