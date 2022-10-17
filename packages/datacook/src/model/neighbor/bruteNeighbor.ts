import { neg, pow, stack, sub, sum, Tensor, Tensor2D, topk, unstack } from "@tensorflow/tfjs-core";
import { checkArray } from "../../utils/validation";
import { DistanceMetric, MetricFactory, MetricName, MetricParams } from "./metrics";
import { NeighborhoodMethod } from "./neighborhood";

export interface BruteNeighborParams {
  metric?: MetricName;
  metricParams?: MetricParams;
}

export class BruteNeighbor implements NeighborhoodMethod {
  protected data: Tensor2D;
  public metric: DistanceMetric;
  constructor(params: BruteNeighborParams = {}) {
    const { metric = 'l2', metricParams = {} } = params;
    this.metric = MetricFactory.getMetric(metric, metricParams);
  }
  public async fit(xData: number[][]): Promise<void> {
    this.data = checkArray(xData, 'float32', 2) as Tensor2D;
  }
  public async query(xData: number[][], k: number, returnDistance = true): Promise<{ indices: number[][]; distances?: number[][] }> {
    const xTensor = checkArray(xData, 'float32', 2) as Tensor2D;
    const rDists: Tensor[] = [];
    unstack(xTensor).map((pt) => {
      const rDist = sum(pow(sub(this.data, pt), this.metric.p), 1);
      rDists.push(rDist);
    });
    const { values, indices } = topk(neg(stack(rDists)), k);
    if (returnDistance) {
      return { indices: indices.arraySync() as number[][], distances: pow(neg(values), 1 / this.metric.p).arraySync() as number[][] };
    } else {
      return { indices: indices.arraySync() as number[][] };
    }
  }
  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    const { data, metric, metricParams } = modelParams;
    this.data = checkArray(data, 'float32', 2) as Tensor2D;
    this.metric = MetricFactory.getMetric(metric, metricParams);
  }
  public async toObject(): Promise<Record<string, any>> {
    const modelParams = {
      data: await this.data.array(),
      metric: this.metric.name,
      metricParams: { p: this.metric.p }
    };
    return modelParams;
  }

}
