import { Tensor } from '../classes';
export abstract class Optimizer {
  params: Tensor[];
  abstract step(): void;
  public zeroGrad(): void {
    this.params.forEach((p: Tensor) => {
      p.zeroGrad();
    });
  }
}
