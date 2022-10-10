export interface NeighborhoodMethod {
  fit: (xData: number[][], params?: any) => void;
  query: (xData: number[][], k: number, returnDistance?: boolean) => { indices: number[][], distances?: number[][] };
}
