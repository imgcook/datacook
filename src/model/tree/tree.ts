import { Tensor1D, sum, equal, add, Tensor, TensorLike, RecursiveArray, gather, slice } from "@tensorflow/tfjs-core";
import { checkArray } from "../../utils/validation";

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
   * Contains the constant prediction value of each node.
   */
  value: number;
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
  weightedNodeSampleCount: number;
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
        importances[node.feature] += node.impurity * node.weightedNodeSampleCount - leftNode.impurity * leftNode.weightedNodeSampleCount - rightNode.impurity * rightNode.weightedNodeSampleCount;
      }
    });
    const sampleCount = this.nodes[0].weightedNodeSampleCount;
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
  public applyDense = (xData: Tensor | RecursiveArray<number>): number => {
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

  public applyDecisionPathDense = (xData: Tensor | RecursiveArray<number>, capacity: Tensor) => {
    const xTensor = checkArray(xData, 'float32', 2);

  }

}

export const buildPrunedTree = (origTree: Tree, leavesInSubTree: boolean[]): Tree => {
} 
