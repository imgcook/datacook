export interface NeighborhoodMethod {
  fit: (xData: number[][], params?: any) => void;
  query: (xData: number[][], k: number, returnDistance?: boolean) => { indices: number[][], distances?: number[][] };
  toObject: () => Record<string, any>;
  fromObject: (modelParams: Record<string, any>) => void;
}
