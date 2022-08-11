import { Vector } from "../../backend-cpu/classes";
import { createZeroVector } from "../../backend-cpu/classes/creation";
import { Matrix, Scalar } from "../classes";
import { Tensor } from "../classes";
import { createZeroMatrix, scalar } from "../classes/creation";
import { add1d, add2d, div1d, div2d, mul1d, mul2d, sub2d } from "../op/binary-op";
import { sqrt1d, sqrt2d, square1d, square2d } from "../op/single-op";
import { Optimizer } from "./classes";

export interface AdaDeltaOptimizerProps {
  learningRate: number;
  rho: number;
  epsilon?: number;
}

export class AdaDeltaOptimizer extends Optimizer {
  protected learningRate: number;
  protected rho: number;
  protected epsilon: number;
  protected accumulatedGrads: Tensor[];
  protected accumulatedUpdate: Tensor[];
  constructor(params: Tensor[], props: AdaDeltaOptimizerProps) {
    super();
    if (!params) {
      throw new TypeError('Params should be provided');
    }
    if (!props) {
      throw new TypeError('Learning rate should be provided');
    }
    const { learningRate, rho, epsilon = 1e-4 } = props;
    this.params = params;
    this.learningRate = learningRate;
    this.rho = rho;
    this.epsilon = epsilon;
    this.accumulatedGrads = new Array(params.length);
    this.accumulatedUpdate = new Array(params.length);
    this.params.forEach((p: Tensor, i: number) => {
      if (p instanceof Matrix) {
        this.accumulatedGrads[i] = createZeroMatrix(p.shape[0], p.shape[1]);
        this.accumulatedUpdate[i] = createZeroMatrix(p.shape[0], p.shape[1]);
      }
      if (p instanceof Vector) {
        this.accumulatedGrads[i] = createZeroVector(p.length);
        this.accumulatedUpdate[i] = createZeroVector(p.length);
      }
      if (p instanceof Scalar) {
        this.accumulatedGrads[i] = scalar(0);
        this.accumulatedUpdate[i] = scalar(0);
      }
    });
  }
  step(): void {
    this.params.forEach((p: Tensor, i: number) => {
      if (p instanceof Matrix) {
        this.accumulatedGrads[i] = add2d(mul2d(this.accumulatedGrads[i] as Matrix, this.rho),
          mul2d(square2d(p.grad), 1 - this.rho));
        const updates = mul2d(div2d(sqrt2d(add2d(this.accumulatedUpdate[i] as Matrix, this.epsilon)),
          sqrt2d(add2d(this.accumulatedGrads[i] as Matrix, this.epsilon))), p.grad);
        this.accumulatedUpdate[i] = add2d(mul2d(this.accumulatedUpdate[i] as Matrix, this.rho),
          mul2d(square2d(updates), 1 - this.rho));
        const val = sub2d(p, mul2d(updates, this.learningRate));
        p.assign(val.values());
      }
      if (p instanceof Vector) {
        this.accumulatedGrads[i] = add1d(mul1d(this.accumulatedGrads[i] as Vector, this.rho),
          mul1d(square1d(p.grad), 1 - this.rho));
        const updates = mul1d(div1d(sqrt1d(add1d(this.accumulatedUpdate[i] as Vector, this.epsilon)),
          sqrt1d(add1d(this.accumulatedGrads[i] as Vector, this.epsilon))), p.grad);
        this.accumulatedUpdate[i] = add1d(mul1d(this.accumulatedUpdate[i] as Vector, this.rho),
          mul1d(square1d(updates), 1 - this.rho));
        const val = add1d(p, mul1d(updates, -this.learningRate));
        p.assign(val.values());
      }
      if (p instanceof Scalar) {
        this.accumulatedGrads[i] = scalar((this.accumulatedGrads[i] as Scalar).data * this.rho +
          Math.pow(p.grad.data, 2) * (1 - this.rho));
        const updates = Math.sqrt((this.accumulatedUpdate[i] as Scalar).data + this.epsilon) /
          Math.sqrt((this.accumulatedGrads[i] as Scalar).data + this.epsilon) * p.grad.data;
        this.accumulatedUpdate[i] = scalar((this.accumulatedUpdate[i] as Scalar).data * this.rho +
          Math.pow(updates, 2) * (1 - this.rho));
        const val = p.values() - updates * this.learningRate;
        p.assign(val);
      }
    });
  }
}
