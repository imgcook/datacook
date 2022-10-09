import { BianryTree } from "./binaryTree";

export class BallTree extends BianryTree {
  protected initNode(iNode: number, startIdx: number, endIdx: number): void {
    const nFeatures = this.dataArr[0].length;
    const centroid = new Array(nFeatures).fill(0);
    // const withSampleWeight = !!this.sampleWeights;
    // TODO: sample weight
    const nPoints = endIdx - startIdx;
    // Deterimin centroids
    for (let i = 0; i < nFeatures; i++) {
      centroid[i] = 0;
    }
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
    return Math.max(0, dist);
  }
  protected minRDist(iNode: number, data: number[]): number {
    const dist = this.rDist(data, this.nodeBounds[iNode]);
    return Math.max(0, dist);
  }
}
