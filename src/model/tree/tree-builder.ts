import { Splitter, SplitRecord } from './splitter';
import { SIZE_MAX, Tree } from './tree';
import { Stack, StackRecord } from './stack';


abstract class TreeBuilder {

}

const INITIAL_STACK_SIZE = 10;
const TREE_UNDEFINED = -2;
const EPSILON = 1e-10;

export class DepthFirstTreeBuilder {
  public splitter: Splitter;
  public minSamplesSplit: number;
  public minSamplesLeaf: number;
  public maxDepth: number;
  public minImpurityDecrease: number;
  public minWeightLeaf: number;

  // public checkInput(X: number[][], y: number[], sampleWeight: number[]): boolean {

  // }
  constructor(splitter: Splitter, minSamplesSplit: number,
    minSamplesLeaf: number,
    minWeightLeaf: number,
    maxDepth: number,
    minImpurityDecrease: number) {

    this.splitter = splitter;
    this.minSamplesSplit = minSamplesSplit;
    this.minWeightLeaf = minWeightLeaf;
    this.maxDepth = maxDepth;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minImpurityDecrease = minImpurityDecrease;
  }

  public build(tree: Tree, X: number[][], y: number[], sampleWeight: number[]): void {
    let initCapacity = 0;
    let maxDepthSeen = -1;

    if (tree.maxDepth <= 10) {
      initCapacity = (Math.pow(2, tree.maxDepth + 1)) - 1;
    } else {
      initCapacity = 2047;
    }
    tree.resize(initCapacity);

    this.splitter.init(X, y, sampleWeight);
    const stack = new Stack(INITIAL_STACK_SIZE);
    let stackRecord: StackRecord;
    let split: SplitRecord;
    const nNodeSamples = this.splitter.nSamples;
    let first = true;
    // push root node to stack
    stack.push({
      start: 0,
      end: nNodeSamples,
      depth: 0,
      parent: TREE_UNDEFINED,
      isLeft: false,
      impurity: Number.MAX_SAFE_INTEGER,
      nConstantFeatures: 0
    });
    while (!stack.isEmpty()) {
      stackRecord = stack.pop();
      const { start, end, depth, parent, isLeft } = stackRecord;
      let { impurity, nConstantFeatures } = stackRecord;
      const nNodeSamples = end - start;
      this.splitter.nodeReset(start, end);
      const weightedNNodeSamples = this.splitter.criterion.weightedNNodeSamples;
      let isLeaf = (depth >= this.maxDepth ||
        nNodeSamples < this.minSamplesSplit ||
        nNodeSamples < 2 * this.minSamplesLeaf ||
        weightedNNodeSamples < 2 * this.minWeightLeaf);
      if (first) {
        impurity = this.splitter.nodeImpurity();
        first = false;
      }
      isLeaf = isLeaf || impurity <= EPSILON;
      if (!isLeaf) {
        const splitInfo = this.splitter.nodeSplit(impurity, nConstantFeatures);
        split = splitInfo.split;
        nConstantFeatures = splitInfo.nConstantFeatures;
        isLeaf = (isLeaf || split.pos >= end || (split.improvement + EPSILON < this.minImpurityDecrease));
      }
      const nodeId = tree.addNode(parent, isLeft, isLeaf, split.feature,
        split.threshold, impurity, nNodeSamples, weightedNNodeSamples);
      if (nodeId === SIZE_MAX) {
        return;
      }
      // this.splitter.no
      if (!isLeaf) {
        // push right node
        stack.push({
          start: split.pos,
          end,
          depth: depth + 1,
          parent: nodeId,
          isLeft: false,
          impurity: split.impurityRight,
          nConstantFeatures
        });
        // push left node
        stack.push({
          start: split.pos,
          end,
          depth: depth + 1,
          parent: nodeId,
          isLeft: true,
          impurity: split.impurityLeft,
          nConstantFeatures
        });
      }
      if (depth > maxDepthSeen) {
        maxDepthSeen = depth;
      }
    }
    tree.resize(tree.nodeCount);
    tree.maxDepth = maxDepthSeen;
  }
}

export const buildPrunedTree = (originTree: Tree, capacity: number): Tree => {
  const tree = new Tree();
};
