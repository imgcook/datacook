import { dispose, RecursiveArray, Tensor } from '@tensorflow/tfjs-core';
import { KMeans } from '.';
import { checkArray } from '../../../utils/validation';

// export class kElbow
export interface InertiaItem {
  inertia: number;
  k: number;
}
export interface KElbowPrams {
  k?: [ number, number ],
  verbose?: boolean
}
export const kElbow = async (kmeans: KMeans, xData: Tensor | RecursiveArray<number>, params: KElbowPrams = {}): Promise<InertiaItem[]> => {
  const xTensor = checkArray(xData);
  const { k = [ 2, 10 ], verbose = false } = params;
  const [ start, end ] = k;
  const elbowsData = [];
  for (let i = start; i < end; i++) {
    kmeans.nClusters = i;
    if (verbose) {
      console.log('Start fitting for nCluster=', i);
    }
    await kmeans.fit(xTensor);
    const inertia = kmeans.inertiaDense(xTensor, kmeans.centroids);
    if (verbose) {
      console.log(`nCluster=${i} fitted, inertia=${inertia}`);
    }
    elbowsData.push({
      inertia,
      k: i
    });
  }
  if (!(xData instanceof Tensor)) {
    dispose(xTensor);
  }
  return elbowsData;
};
