import { BianryTree } from "./binaryTree";
import { NeighborhoodMethod } from "./neighborhood";

export class BallTree extends BianryTree implements NeighborhoodMethod {
  public nodeBounds: number[][];

  protected initNodeBounds(): void {
    this.nodeBounds = new Array(this.nNodes);
    for (let i = 0; i < this.nNodes; i++) {
      this.nodeBounds[i] = new Array(this.nFeatures).fill(0);
    }
  }
  public async fit(data: number[][]): Promise<void> {
    await super.fit(data);
  }
  protected initNode(iNode: number, startIdx: number, endIdx: number): void {
    const nFeatures = this.dataArr[0].length;
    const centroid = new Array(nFeatures).fill(0);
    // const withSampleWeight = !!this.sampleWeights;
    // TODO: sample weight
    const nPoints = endIdx - startIdx;
    // Deterimin centroids
    for (let i = startIdx; i < endIdx; i++) {
      const dp = this.dataArr[i];
      for (let j = 0; j < nFeatures; j++) {
        centroid[j] += dp[j];
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      centroid[j] /= nPoints;
    }
    this.nodeBounds[iNode] = centroid;
    // determine radius
    let radius = 0;
    for (let i = startIdx; i < endIdx; i++) {
      radius = Math.max(radius, this.rDist(centroid, this.dataArr[this.idxArr[i]]));
    }
    this.nodeDataArr[iNode].startIdx = startIdx;
    this.nodeDataArr[iNode].endIdx = endIdx;
    this.nodeDataArr[iNode].radius = radius;
    return;
  }

  protected minDist(iNode: number, data: number[]): number {
    const dist = this.dist(data, this.nodeBounds[iNode]);
    return Math.max(0, dist - this.nodeDataArr[iNode].radius);
  }

  // TODO: reduced distance
  protected minRDist(iNode: number, data: number[]): number {
    const dist = this.dist(data, this.nodeBounds[iNode]);
    return this.metric.distToRDist(Math.max(0, dist - this.nodeDataArr[iNode].radius));
  }

  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    super.fromObject(modelParams);
    this.nodeBounds = modelParams.nodeBounds;
  }

  public async toObject(): Promise<Record<string, any>> {
    const modelParams = await super.toObject();
    modelParams.nodeBounds = this.nodeBounds;
    return modelParams;
  }
}
