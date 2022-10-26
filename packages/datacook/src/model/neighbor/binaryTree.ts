import { NodeIndex } from "@tensorflow/tfjs-layers/dist/keras_format/node_config";
import { checkJSArray } from "../../utils/validation";
import { NeighborHeap } from "./heap";
import { NeighborhoodMethod } from "./neighborhood";
import { quickPartitionNode } from './utils';
import { MetricName, DistanceMetric, MetricFactory, MetricParams } from './metrics';
import { convertMat2Csr, convertCsr2Mat } from '../../utils/csr';

export interface BinaryTreeNode {
  isLeaf?: boolean;
  startIdx: number;
  endIdx: number;
  radius?: number;
}

export interface BinaryTreeParams {
  metric?: MetricName;
  metricParams?: MetricParams;
  leafSize?: number;
  sampleWeights?: number[];
}

export abstract class BianryTree implements NeighborhoodMethod {
  // data samples
  public dataArr: number[][];
  // sample weights
  public sampleWeights: number[];
  // node data
  public nodeDataArr: BinaryTreeNode[];
  public idxArr: number[];
  public metric: DistanceMetric;
  // public nodeBounds: number[][];
  // maximum number of leaves
  public leafSize: number;
  protected nFeatures: number;
  // number of nodes
  protected nNodes: number;
  protected nLevels: number;

  protected abstract initNodeBounds(): void;

  constructor(params: BinaryTreeParams = {}) {
    const { metric = 'l2', metricParams = {}, leafSize = 40 } = params;
    this.metric = MetricFactory.getMetric(metric, metricParams);
    this.leafSize = leafSize;
  }

  public async fit(data: number[][]): Promise<void> {
    this.dataArr = checkJSArray(data, 'float32', 2) as number[][];
    if (this.dataArr.length === 0) {
      throw new TypeError('Input is empry array');
    }
    const nSamples = this.dataArr.length;
    const nFeatures = this.dataArr[0].length;

    // Determine leaf size and level size
    this.nLevels = Math.ceil(Math.log2(Math.max(1, (nSamples - 1) / this.leafSize)));
    this.nNodes = Math.pow(2, this.nLevels) - 1;
    this.idxArr = new Array(nSamples).fill(0).map((d, i) => i);
    this.nodeDataArr = new Array(this.nNodes);
    this.nFeatures = nFeatures;
    for (let i = 0; i < this.nNodes; i++) {
      this.nodeDataArr[i] = { startIdx: -1, endIdx: -1 };
    }
    this.initNodeBounds();
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
    // return Math.sqrt(x1.map((d, i) => Math.pow(d - x2[i], 2)).reduce((a, b) => a + b));
    return this.metric.dist(x1, x2);
  }
  /**
   * redueced distance between x1 and x2
   */
  public rDist(x1: number[], x2: number[]): number {
    /**
    * reduced euclidean distance
    */
    return this.metric.rDist(x1, x2);
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
      // console.log('iter', i);
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
        const reducedDistR = this.minRDist(i2, data);

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

  public async toObject(): Promise<Record<string, any>> {
    const modelParams: Record<string, any> = {};
    modelParams.dataArr = convertMat2Csr(this.dataArr);
    modelParams.sampleWeights = this.sampleWeights;
    modelParams.nodeDataArr = this.nodeDataArr;
    modelParams.idxArr = this.idxArr;
    modelParams.leafSize = this.leafSize;
    modelParams.nFeatures = this.nFeatures;
    modelParams.nNodes = this.nNodes;
    modelParams.nLevels = this.nLevels;
    modelParams.metric = this.metric.name;
    modelParams.metricPrams = { p: this.metric.p };
    return modelParams;
  }

  public async fromObject(modelParams: Record<string, any>): Promise<void> {
    const {
      dataArr,
      sampleWeights,
      nodeDataArr,
      idxArr,
      leafSize,
      nFeatures,
      nNodes,
      nLevels,
      metric,
      metricParams
    } = modelParams;
    this.dataArr = convertCsr2Mat(dataArr);
    this.sampleWeights = sampleWeights;
    this.nodeDataArr = nodeDataArr;
    this.idxArr = idxArr;
    this.leafSize = leafSize;
    this.nFeatures = nFeatures;
    this.nNodes = nNodes;
    this.nLevels = nLevels;
    this.metric = MetricFactory.getMetric(metric, metricParams);
  }
}
