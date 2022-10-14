import { neg, pow, stack, sub, sum, Tensor, Tensor2D, topk, unstack } from "@tensorflow/tfjs-core";
import { checkArray } from "../../utils/validation";
import { NeighborhoodMethod } from "./neighborhood";

export class BruteNeighbor implements NeighborhoodMethod {
  protected data: Tensor2D;
  public async fit(xData: number[][]): Promise<void> {
    this.data = checkArray(xData, 'float32', 2) as Tensor2D;
  }
  public async query(xData: number[][], k: number, returnDistance = true): Promise<{ indices: number[][]; distances?: number[][] }> {
    const xTensor = checkArray(xData, 'float32', 2) as Tensor2D;
    const rDists: Tensor[] = [];
    unstack(xTensor).map((pt) => {
      const rDist = sum(pow(sub(this.data, pt), 2), 1);
      rDists.push(rDist);
    });
    console.log(rDists[0].shape);
    rDists[1].print();
    rDists[2].print();
    const { values, indices } = topk(neg(stack(rDists)), k);
    if (returnDistance) {
      return { indices: indices.arraySync() as number[][], distances: neg(values).arraySync() as number[][] };
    } else {
      return { indices: indices.arraySync() as number[][] };
    }
  }
  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    const { data } = modelParams;
    this.data = checkArray(data, 'float32', 2) as Tensor2D;
  }
  public async toObject(): Promise<Record<string, any>> {
    const modelParams = {
      data: await this.data.array()
    };
    return modelParams;
  }

}
