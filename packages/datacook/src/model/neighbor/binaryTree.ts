import { Tensor2D } from "@tensorflow/tfjs-core";
import { NodeIndex } from "@tensorflow/tfjs-layers/dist/keras_format/node_config";
import { checkJSArray } from "../../utils/validation";
import { NeighborHeap } from "./heap";
import { quickPartitionNode } from './utils';

export type BinaryTreeMetrics = "minkowski" | "";

export interface BinaryTreeParams {
  leafSize?: number;
  metrics?: BinaryTreeMetrics;
  sampleWeights?: number[];
}

export interface BinaryTreeNode {
  isLeaf?: boolean;
  startIdx: number;
  endIdx: number;
  radius?: number;
}

export abstract class BianryTree {
  // data samples
  public dataArr: number[][];
  // sample weights
  public sampleWeights: number[];
  // node data
  public nodeDataArr: BinaryTreeNode[];
  public idxArr: number[];
  public nodeBounds: number[][];
  // maximum number of leaves
  public leafSize: number;
  // number of nodes
  public nNodes: number;
  public nLevels: number;


  public async fit(data: number[][], params: BinaryTreeParams = {}): Promise<void> {
    const { leafSize = 40, metrics = "minkowski" } = params;
    this.dataArr = checkJSArray(data, 'float32', 2) as number[][];
    if (this.dataArr.length === 0) {
      throw new TypeError('Input is empry array');
    }
    const nSamples = this.dataArr.length;
    const nFeatures = this.dataArr[0].length;

    // Determine leaf size and level size
    this.leafSize = leafSize;
    this.nLevels = Math.ceil(Math.log2(Math.max(1, (nSamples - 1) / this.leafSize)));
    this.nNodes = Math.pow(2, this.nLevels) - 1;
    this.idxArr = new Array(nSamples).fill(0).map((d, i) => i);
    this.nodeDataArr = new Array(this.nNodes);
    this.nodeBounds = new Array(this.nNodes);
    for (let i = 0; i < this.nNodes; i++) {
      this.nodeDataArr[i] = { startIdx: -1, endIdx: -1 };
      this.nodeBounds[i] = new Array(nFeatures).fill(0);
    }
    this.recursiveBuild(0, 0, this.dataArr.length);
    // this.recursiveBuild();
  }

  /**
   * Minimum reduced distance between a point and a node
   */
  protected abstract minRDist(iNode: number, data: number[]): number;

  /**
   * Minimum distance between a point and a node
   */
  protected abstract minDist(iNode: number, data: number[]): number;
  protected abstract initNode(iNode: NodeIndex, startIdx: number, endIdx: number): void;

  /**
   * Find the dimension with the largest spread
   * @param data input data
   * @param idxArr index array
   * @param nFeatures number of features
   * @param nPoints number of data points
   */
  protected findNodeSplitDim(data: number[][], idxArr: number[], nFeatures: number, nPoints: number): number {
    let maxSpread = 0;
    let idxMax = 0;
    for (let i = 0; i < nFeatures; i++) {
      let maxVal = data[idxArr[0]][i];
      let minVal = maxVal;
      for (let j = 0; j < nPoints; j++) {
        const val = data[idxArr[j]][i];
        maxVal = Math.max(maxVal, val);
        minVal = Math.min(minVal, val);
      }
      const spread = maxVal - minVal;
      if (spread > maxSpread) {
        maxSpread = spread;
        idxMax = i;
      }
    }
    return idxMax;
  }
  /**
   * Build tree recursively
   * @param iNode node for current step
   * @param startIdx satrt index
   * @param endIdx end index
   */
  protected recursiveBuild(iNode: number, startIdx: number, endIdx: number): void {
    const nFeatures = this.dataArr[0].length;
    const nPoints = endIdx - startIdx;
    this.initNode(iNode, startIdx, endIdx);
    if (2 * iNode + 1 >= this.nNodes) {
      // this.
      this.nodeDataArr[iNode].isLeaf = true;
      if (endIdx - startIdx > 2 * this.leafSize) {
        console.warn('Internal: memory layout is flawed: not enough nodes allocated');
      }
    } else {
      if (endIdx - startIdx < 2) {
        this.nodeDataArr[iNode].isLeaf = true;
      } else {
        const nMid = Math.floor((endIdx - startIdx) / 2);
        const iMax = this.findNodeSplitDim(this.dataArr, this.idxArr, nFeatures, nPoints);
        quickPartitionNode(this.dataArr, this.idxArr, startIdx, endIdx, iMax, nMid);
        this.recursiveBuild(2 * iNode + 1, startIdx, startIdx + nMid);
        this.recursiveBuild(2 * iNode + 2, startIdx + nMid, endIdx);
      }
    }
  }
  protected dist(x1: number[], x2: number[]): number {
    return Math.sqrt(x1.map((d, i) => Math.pow(d - x2[i], 2)).reduce((a, b) => a + b));
  }
  /**
   * redueced distance between x1 and x2
   */
  public rDist(x1: number[], x2: number[]): number {
    /**
    * reduced euclidean distance
    */
    return x1.map((d, i) => Math.pow(d - x2[i], 2)).reduce((a, b) => a + b);
  }

  /**
   * Query the tree for n nearest neighbors
   * @param xData input data
   * @param k number of nearest neighbors
   * @param returnDistance if distance would be returned
   */
  public query(xData: number[][], k: number, returnDistance = true): { indices: number[][], distances?: number[][] } {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    if (!xArray.length) {
      throw new TypeError("Empty Input");
    }
    if (xArray[0].length !== this.dataArr[0].length) {
      throw new TypeError("Query data dimension must match training data dimension");
    }
    if (k > this.dataArr.length) {
      throw new TypeError("k must be less than or equal to the number of training points");
    }
    const heap = new NeighborHeap(xArray.length, k);
    for (let i = 0; i < xArray.length; i++) {
      const reducedDist = this.minRDist(0, xArray[i]);
      this.querySingleDepthFirst(0, xArray[i], i, heap, reducedDist);
    }
    if (returnDistance) {
      return { indices: heap.indices, distances: heap.distances };
    } else {
      return { indices: heap.indices };
    }
  }
  /**
   * Query single data point
   */
  public querySingleDepthFirst(iNode: number, data: number[], iPt: number, heap: NeighborHeap, dist: number): void {
    const nodeInfo = this.nodeDataArr[iNode];
    if (dist > heap.distances[iPt][0]) {
      return;
    } else {
      if (nodeInfo.isLeaf) {
        this.nLevels += 1;
        for (let i = nodeInfo.startIdx; i < nodeInfo.endIdx; i++) {
          const distPt = this.rDist(data, this.dataArr[this.idxArr[i]]);
          if (!heap.distances.length || distPt < heap.distances[iPt][0]) {
            heap.push(iPt, distPt, this.idxArr[i]);
          }
        }
      } else {
        const i1 = 2 * iNode + 1;
        const i2 = i1 + 1;
        const reducedDistL = this.minRDist(i1, data);
        const reducedDistR = this.minDist(i2, data);

        if (reducedDistL <= reducedDistR) {
          this.querySingleDepthFirst(i1, data, iPt, heap, reducedDistL);
          this.querySingleDepthFirst(i2, data, iPt, heap, reducedDistR);
        } else {
          this.querySingleDepthFirst(i2, data, iPt, heap, reducedDistR);
          this.querySingleDepthFirst(i1, data, iPt, heap, reducedDistL);
        }
      }
    }
  }
}
