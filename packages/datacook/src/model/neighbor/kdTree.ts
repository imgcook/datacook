import { BianryTree } from "./binaryTree";
import { NeighborhoodMethod } from "./neighborhood";

export class KDTree extends BianryTree implements NeighborhoodMethod {
  protected lowerBounds: number[][];
  protected uppperBounds: number[][];
  protected initNodeBounds(): void {
    this.lowerBounds = new Array(this.nNodes);
    this.uppperBounds = new Array(this.nNodes);
    for (let i = 0; i < this.nNodes; i++) {
      this.lowerBounds[i] = new Array(this.nFeatures).fill(Number.MAX_SAFE_INTEGER);
      this.uppperBounds[i] = new Array(this.nFeatures).fill(Number.MIN_SAFE_INTEGER);
    }
  }
  protected initNode(iNode: number, startIdx: number, endIdx: number): void {
    const lowerBound = this.lowerBounds[iNode];
    const uppperBound = this.uppperBounds[iNode];
    for (let i = startIdx; i < endIdx; i++) {
      for (let j = 0; j < this.nFeatures; j++) {
        lowerBound[j] = Math.min(lowerBound[j], this.dataArr[this.idxArr[i]][j]);
        uppperBound[j] = Math.max(uppperBound[j], this.dataArr[this.idxArr[i]][j]);
      }
    }
    const radius = this.dist(uppperBound, lowerBound);
    this.nodeDataArr[iNode].startIdx = startIdx;
    this.nodeDataArr[iNode].endIdx = endIdx;
    this.nodeDataArr[iNode].radius = radius;
  }
  protected minDist(iNode: number, data: number[]): number {
    return this.metric.rDistToDist(this.minRDist(iNode, data));
  }
  protected minRDist(iNode: number, data: number[]): number {
    let rDist = 0;
    for (let j = 0; j < this.nFeatures; j++) {
      const dLow = this.lowerBounds[iNode][j] - data[j];
      const dHigh = data[j] - this.uppperBounds[iNode][j];
      const dist = Math.max(0, dLow) + Math.max(0, dHigh);
      rDist += this.metric.distToRDist(dist);
    }
    return rDist;
  }

  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    super.fromObject(modelParams);
    this.uppperBounds = modelParams.uppperBounds;
    this.lowerBounds = modelParams.lowerBounds;
  }

  public async toObject(): Promise<Record<string, any>> {
    const modelParams = await super.toObject();
    modelParams.uppperBounds = this.uppperBounds;
    modelParams.lowerBounds = this.lowerBounds;
    return modelParams;
  }
}
