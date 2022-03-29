import { Tree } from "./tree";


abstract class PruneController {
  abstract stopPruning(effectiveAlpha: number): boolean;
  // abstract saveMetrics(effectiveAlpha: number, subtreeImpurities: number): void;
  abstract afterPruning(inSubtree: boolean[]): void;
}

export class AlphaPruner extends PruneController {
  public alpha: number;
  public capacity: number;

  constructor(alpha: number) {
    super();
    this.alpha = alpha;
    this.capacity = 0;
  }

  public stopPruning(effectiveAlpha: number): boolean {
    return this.alpha < effectiveAlpha;
  }

  public afterPruning(inSubtree: boolean[]): void {
    for (let i = 0; i < inSubtree.length; i++) {
      if (inSubtree[i]) {
        this.capacity += 1;
      }
    }
  }
}

export const costComplexityPrune = (origTree: Tree, controller: PruneController): boolean[] => {
  const nNodes = origTree.nodeCount;
  const weightedNNodeSamples = origTree.nodes.map((n) => n.weightedNNodeSamples);
  const totalSumWeights = weightedNNodeSamples.reduce((a, b) => a + b);
  const nodes = origTree.nodes;
  // weiwghted impurity
  const rNode = new Array(nNodes).fill(0);
  const rBranch = new Array(nNodes).fill(0);
  const nLeaves = new Array(nNodes).fill(0);
  const leavesInSubTree: boolean[] = new Array(nNodes).fill(false);
  const candidateNodes: boolean[] = new Array(nNodes).fill(false);
  const inSubTree: boolean[] = new Array(nNodes).fill(true);
  const stack: Array<[ number, number ]> = [];

  for (let i = 0; i < nNodes; i++) {
    rNode[i] = weightedNNodeSamples[i] * nodes[i].impurity / totalSumWeights;
  }
  stack.push([ 0, -2 ]);
  while (stack.length) {
    const stackRecord = stack.pop();
    const nodeIdx = stackRecord[0];
    const leftChildIdx = nodes[nodeIdx].leftChild;
    const rightChildIdx = nodes[nodeIdx].rightChild;
    if (nodes[nodeIdx].leftChild === -1) {
      leavesInSubTree[nodeIdx] = true;
    } else {
      stack.push([ leftChildIdx, nodeIdx ]);
      stack.push([ rightChildIdx, nodeIdx ]);
    }
  }

  for (let i = 0; i < nNodes; i++) {
    if (!leavesInSubTree[i]) {
      continue;
    }
    const currentR = rNode[i];
    let j = i;
    // bubble up to ancesctor nodes
    while (j > 0) {
      const parentIdx = nodes[j].parent;
      rBranch[parentIdx] += currentR;
      nLeaves[parentIdx] += 1;
      j = parentIdx;
    }
  }

  for (let i = 0; i < leavesInSubTree.length; i++) {
    candidateNodes[i] = !leavesInSubTree[i];
  }
  while (candidateNodes[0]) {
    let effectiveAlpha = Number.MAX_SAFE_INTEGER;
    let subTreeAlpha = Number.MAX_SAFE_INTEGER;
    let prunedBranchNodeIdx = -1;
    for (let i = 0; i < nNodes; i++) {
      if (!candidateNodes[i]) {
        continue;
      }
      subTreeAlpha = (rNode[i] - rBranch[i]) / nLeaves[i];
      if (subTreeAlpha < effectiveAlpha) {
        effectiveAlpha = subTreeAlpha;
        prunedBranchNodeIdx = i;
      }
    }
    if (controller.stopPruning(effectiveAlpha)) {
      break;
    }
    stack.push([ prunedBranchNodeIdx, 0 ]);
    while (stack.length) {
      const nodeIdx = stack.pop()[0];
      // already been marked
      if (!inSubTree[nodeIdx]) {
        continue;
      }
      candidateNodes[nodeIdx] = false;
      leavesInSubTree[nodeIdx] = false;
      inSubTree[nodeIdx] = false;
      if (nodes[nodeIdx].leftChild !== -1) {
        stack.push([ nodes[nodeIdx].leftChild, 0 ]);
        stack.push([ nodes[nodeIdx].rightChild, 0 ]);
      }
    }
    leavesInSubTree[prunedBranchNodeIdx] = true;
    inSubTree[prunedBranchNodeIdx] = true;

    const nPrunedLeaves = nLeaves[prunedBranchNodeIdx] - 1;
    nLeaves[prunedBranchNodeIdx] = 0;

    const rDiff = rNode[prunedBranchNodeIdx] - rBranch[prunedBranchNodeIdx];
    rBranch[prunedBranchNodeIdx] = rNode[prunedBranchNodeIdx];

    let nodeIdx = nodes[prunedBranchNodeIdx].parent;
    // bubble up to update ancestors
    while (nodeIdx >= 0) {
      nLeaves[nodeIdx] -= nPrunedLeaves;
      rBranch[nodeIdx] += rDiff;
      nodeIdx = nodes[nodeIdx].parent;
    }
  }
  controller.afterPruning(inSubTree);
  return leavesInSubTree;
};

const buildPrunedTreeByLeaves = (originTree: Tree, leavesInSubTree: boolean[]): Tree => {
  // stack of [ originNodeIndex, depth, parentIndex, isLeft ]
  const stack: Array<[ number, number, number, boolean]> = [];
  const nodes = originTree.nodes;
  const tree = new Tree(originTree.nFeature);
  stack.push([ 0, 0, -1, false ]);
  while (stack.length) {
    const [ originNodeIdx, depth, parentIdx, isLeft ] = stack.pop();
    const node = nodes[originNodeIdx];
    const isLeaf = leavesInSubTree[originNodeIdx];
    const newNodeId = tree.addNode(
      parentIdx,
      isLeft,
      isLeaf,
      node.feature,
      node.threshold,
      node.impurity,
      node.nNodeSamples,
      node.weightedNNodeSamples,
      node.value);
    if (!isLeaf && node.leftChild !== -1) {
      stack.push([ node.leftChild, depth + 1, newNodeId, true ]);
    }
    if (!isLeaf && node.rightChild !== -1) {
      stack.push([ node.rightChild, depth + 1, newNodeId, false ]);
    }
  }
  return tree;
};

export const buildPrunedTree = (originTree: Tree, ccpAlpha: number): Tree => {
  const controller = new AlphaPruner(ccpAlpha);
  const leavesInSubTree = costComplexityPrune(originTree, controller);
  const tree = buildPrunedTreeByLeaves(originTree, leavesInSubTree);
  return tree;
};
