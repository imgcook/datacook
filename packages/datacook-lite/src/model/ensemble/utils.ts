import { DecisionTreeRegressorPredictor } from "../tree/decision-tree-predictor";
import { Node } from "../tree/tree";

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
    if (node.value) {
      out[i][k] += scale * node.value[0];
    }
  }
};

export const predictStages = (x: number[][], estimators: DecisionTreeRegressorPredictor[], scale: number, out: number[][]): number[][] => {
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

