import { DistanceMetric } from "./metrics";

export interface NeighborhoodMethod {
  metric: DistanceMetric;
  fit: (xData: number[][], params?: any) => void;
  query: (xData: number[][], k: number, returnDistance?: boolean) => Promise<{ indices: number[][], distances?: number[][] }> | { indices: number[][], distances?: number[][] };
  toObject: () => Record<string, any>;
  fromObject: (modelParams: Record<string, any>) => void;
}
