import { checkJSArray } from "../../utils/validation";

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
  // /**
  //  * Counts of samples of each classes reaching node i
  //  */
  // nClassSamples?: number[];
  /**
   * Contains the constant prediction value of each node.
   */
  value?: number[];
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
  /**
   * number of training samples reaching node i
   */
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
  // class count
  public nClass: number;
  // nodes
  public nodes: Node[];
  public nOutput: number;

  constructor (nFeature?: number, nClass?: number) {
    this.nFeature = nFeature;
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
  public applyDense = (xData: number[][]): number[][] => {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const nSamples = xArray.length;
    const values = [];
    for (let i = 0; i < nSamples; i++) {
      const sample = xArray[i];
      let node = this.nodes[0];
      while (node.leftChild != -1 && node.rightChild != -1) {
        const xFeatureVal = sample[node.feature];
        if (xFeatureVal <= node.threshold) {
          node = this.nodes[node.leftChild];
        } else {
          node = this.nodes[node.rightChild];
        }
      }
      values.push(node.value);
    }
    return values;
  };
  /**
   * Finds the terminal region (=leaf node) for each sample in X.
   * @param xData input sample
   */
  public applyNode = (xData: number[][]): number[] => {
    const xArray = checkJSArray(xData, 'float32', 2) as number[][];
    const nSamples = xArray.length;
    const leaves = [];
    for (let i = 0; i < nSamples; i++) {
      const sample = xArray[i];
      let node = this.nodes[0];
      let curNode = 0;
      while (node.leftChild != -1 && node.rightChild != -1) {
        const xFeatureVal = sample[node.feature];
        if (xFeatureVal <= node.threshold) {
          node = this.nodes[node.leftChild];
          if (node.leftChild !== -1) {
            curNode = node.leftChild;
          }
        } else {
          node = this.nodes[node.rightChild];
          if (node.rightChild !== -1) {
            curNode = node.rightChild;
          }
        }
      }
      leaves.push(curNode);
    }
    return leaves;
  };

  /**
   * Find the decsision path for each sample in X
   * @param xData
   */
  public decisionPathDense = (xData: number[][]): number[][] => {
    const nSamples = xData.length;
    const path: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const curPath: number[] = [];
      const sample = xData[i];
      let node = this.nodes[0];
      curPath.push(0);
      while (node.leftChild != -1 && node.rightChild != -1) {
        const xFeatureVal = sample[node.feature];
        if (xFeatureVal <= node.threshold) {
          node = this.nodes[node.leftChild];
          curPath.push(node.leftChild);
        } else {
          node = this.nodes[node.rightChild];
          curPath.push(node.rightChild);
        }
      }
      path.push(curPath);
    }
    return path;
  }

  public predict(x: number[][]): number[][] {
    return this.applyDense(x);
  }

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

  public fromJson(modelJson: string): void {
    const params = JSON.parse(modelJson);
    const { nodeCount, capacity, maxDepth, nFeature, nOutput, nClass, nodes } = params;
    this.nodeCount = nodeCount ? nodeCount : this.nodeCount;
    this.capacity = capacity ? capacity : this.capacity;
    this.maxDepth = maxDepth ? maxDepth : this.maxDepth;
    this.nFeature = nFeature ? nFeature : this.nFeature;
    this.nOutput = nOutput ? nOutput : this.nOutput;
    this.nClass = nClass ? nClass : this.nClass;
    this.nodes = nodes ? nodes : this.nodes;
  }

  public toJson(): string {
    const params = {
      nodeCount: this.nodeCount,
      capacity: this.capacity,
      maxDepth: this.maxDepth,
      nFeature: this.nFeature,
      nOutput: this.nOutput,
      nClass: this.nClass,
      nodes: this.nodes
    };
    return JSON.stringify(params);
  }

  public addNode(parent: number,
    isLeft: boolean,
    isLeaf: boolean,
    feature: number,
    threshold: number,
    impurity: number,
    nNodeSamples: number,
    weightedNNodeSamples: number,
    value: number[]
  ): number {
    const nodeId = this.nodeCount;
    if (nodeId >= this.capacity) {

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
      isLeft: isLeft,
      value: value
    };

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
