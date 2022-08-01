import { DecisionTreeRegressor } from "../tree";
import { Node } from "../tree/tree";

export const randomSampleMask = (nTotalSamples: number, nTotalInBag: number): number[] => {
  const sampleMask = new Array(nTotalSamples).fill(0);
  let nBagged = 0;
  for (let i = 0; i < nTotalSamples; i++) {
    const rand = Math.random();
    if (rand * (nTotalSamples - i) < nTotalInBag - nBagged) {
      sampleMask[i] = 1;
      nBagged += 1;
    }
  }
  return sampleMask;
};

export const getInverseMask = (sampleMask: number[]): number[] => {
  return sampleMask.map((d: number) => d ? 0 : 1);
};

export const predictRegressionTreeFast = (x: number[][], nodes: Node[], k: number, out: number[][], scale: number): void => {
  for (let i = 0; i < x.length; i++) {
    let node = nodes[0];
    while (node.leftChild != -1) {
      if (x[i][node.feature] <= node.threshold) {
        node = nodes[node.leftChild];
      } else {
        node = nodes[node.rightChild];
      }
    }
    out[i][k] += scale * node.value[0];
  }
};

export const predictStages = (x: number[][], estimators: DecisionTreeRegressor[], scale: number, out: number[][]): number[][] => {
  const k = estimators.length;
  // for (let i = 0; i < x.length; i++) {
  //   out.push(new Array(k).fill(0));
  // }
  for (let i = 0; i < k; i++) {
    const nodes = estimators[i].tree.nodes;
    predictRegressionTreeFast(x, nodes, i, out, scale);
  }
  return out;
};

// export const predictStage = (x: number[][], estimator)
