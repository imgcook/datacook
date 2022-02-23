import { Tensor1D, sum, equal, add, Tensor, TensorLike, RecursiveArray, gather, slice } from "@tensorflow/tfjs-core";
import { checkArray } from "../../utils/validation";

export const SIZE_MAX = Number.MAX_SAFE_INTEGER;
export interface Node {
  parent: number;
  /**
   * leftChild holds the node id of the left child of node i.
   */
  leftChild: number;
  /**
   * rightChild holds the node id of the right child of node i.
   */
  rightChild: number;
  /**
   * impurity holds the impurity (i.e., the value of the splitting criterion) at node i.
   */
  impurity: number;
  /**
   * Counts of each class
   */
  counts?: number;

  /**
   * Contains the constant prediction value of each node.
   */
  value?: number;
  /**
   * Feature holds the feature to split on, for the internal node.
   */
  feature: number;
  /**
   * Threshold holds the threshold for the internal node.
   */
  threshold: number;
  /**
   * weighted_n_node_samples holds the weighted number of training samples reaching node.
   */
  weightedNNodeSamples: number;
  nNodeSamples: number;
  isLeft: boolean;
}

export interface TreeParams {
  nFeature: number;
  nOutput: number;
}

export class Tree {
  // The number of nodes (internal nodes + leaves) in the tree.
  public nodeCount: number;
  // The current capacity (i.e., size) of the arrays, which is at least as great as `node_count`.
  public capacity: number;
  // The depth of the tree, i.e. the maximum depth of its leaves.
  public maxDepth: number;
  // Feature count
  public nFeature: number;
  // output count
  public nOutput: number;
  // class count
  public nClass: number;
  // nodes
  public nodes: Node[];

  constructor (nFeature: number, nOutput: number, nClass: number) {
    this.nFeature = nFeature;
    this.nOutput = nOutput;
    this.nClass = nClass;
    this.nodes = [];
    this.nodeCount = 0;
  }

  public nLeaves = (): number => {
    let n = 0;
    this.nodes.forEach((node) => { n += Number(node.leftChild == -1 && node.rightChild == -1); });
    return n;
  };

  public computeFeatureImportance = (normalize = true): number[] => {
    let importances = new Array(this.nFeature).fill(0);
    this.nodes.forEach((node) => {
      if (node.leftChild != -1 && node.rightChild != -1){
        const leftNode = this.nodes[node.leftChild];
        const rightNode = this.nodes[node.rightChild];
        importances[node.feature] += node.impurity * node.weightedNNodeSamples - leftNode.impurity * leftNode.weightedNNodeSamples - rightNode.impurity * rightNode.weightedNNodeSamples;
      }
    });
    const sampleCount = this.nodes[0].weightedNNodeSamples;
    importances = importances.map((imp) => imp * 1.0 / sampleCount);
    if (normalize) {
      const normalizer = importances.reduce((sum, imp) => sum + imp);
      if (normalizer > 0) {
        importances = importances.map((imp) => imp / normalizer);
      }
    }
    return importances;
  };

  /**
   * Finds the terminal region (=leaf node) for each sample in X.
   * @param xData input sample
   */
  public applyDense = (xData: number[][]): number => {
    const xTensor = checkArray(xData, 'float32', 2);
    const nSamples = xTensor.shape[0];
    for (let i = 0; i < nSamples; i++) {
      const sample = gather(xData, i);
      let node = this.nodes[0];
      while (node.leftChild != -1 && node.rightChild != -1) {
        const xFeatureVal = slice(sample, node.feature, 1).dataSync()[0];
        if (xFeatureVal <= node.threshold) {
          node = this.nodes[node.leftChild];
        } else {
          node = this.nodes[node.rightChild];
        }
      }
      return node.value;
    }
  };

  public resize(capacity: number): void {
    if (capacity === this.capacity && this.nodes) {
      return;
    }
    if (capacity === SIZE_MAX) {
      if (this.capacity === 0) {
        this.capacity = 3;
      } else {
        this.capacity = 2 * this.capacity;
      }
    } else {
      this.capacity = capacity;
    }
    if (capacity < this.nodeCount) {
      this.nodeCount = capacity;
    }
    // TODO: capacity > this.nodeCount
  }
  // public applyDecisionPathDense(xData: Tensor | RecursiveArray<number>, capacity: Tensor): void {
  //   const xTensor = checkArray(xData, 'float32', 2);
  // }
  public addNode(parent: number,
    isLeft: boolean,
    isLeaf: boolean,
    feature: number,
    threshold: number,
    impurity: number,
    nNodeSamples: number,
    weightedNNodeSamples: number
  ): number {
    const nodeId = this.nodeCount;
    if (nodeId >= this.capacity) {
      // if (this.resize())
      this.resize(nodeId);
    }
    const node = {
      parent: parent,
      impurity: impurity,
      nNodeSamples: nNodeSamples,
      weightedNNodeSamples: weightedNNodeSamples,
      leftChild: -1,
      rightChild: -1,
      feature: !isLeaf ? feature : -1,
      threshold: !isLeaf ? threshold : -1,
      isLeft: isLeft
    };
    // node.impurity = impurity;
    // node.nNodeSamples = nNodeSamples;
    // node.weightedNNodeSamples = weightedNNodeSamples;

    // if (!isLeaf) {
    //   node.feature = feature;
    //   node.threshold = threshold;
    // }
    // if (isLeaf) {
    //   node.leftChild = null;
    //   node.rightChild = null;
    //   node.feature = null;
    //   node.threshold = null;
    // } else {
    //   node.feature = feature;
    //   node.threshold = threshold;
    // }
    if (parent >= 0) {
      if (isLeft) {
        this.nodes[parent].leftChild = nodeId;
      } else {
        this.nodes[parent].rightChild = nodeId;
      }
    }
    this.nodes.push(node);
    this.nodeCount += 1;
    return nodeId;

  }
}

// export const buildPrunedTree = (origTree: Tree, leavesInSubTree: boolean[]): Tree => {
// };
