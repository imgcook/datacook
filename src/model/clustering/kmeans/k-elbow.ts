import { dispose, RecursiveArray, Tensor } from '@tensorflow/tfjs-core';
import { KMeans } from '.';
import { checkArray } from '../../../utils/validation';

// export class kElbow
export interface InertiaItem {
  inertia: number;
  k: number;
  model: KMeans;
}
export interface KElbowPrams {
  k?: [ number, number ],
  verbose?: boolean
}

/**
 * K-elbow implements the “elbow” method to help users to select the optimal number of clusters
 * by fitting the model with a range of values for K
 * @param kmeans original kmeans model
 * @param xData input data
 * @param params paramters
 * Option in parameters
 * -----
 * - `k`: [ start<number>, end<number> ], start for minimum number of clusters, end for maximum number of clusters,
 * **default=[ 2, 10 ]**
 * - `verbose`: boolean, verbosity, **default=false**
 * @returns Array of inertia for each of clusters
 */
export const kElbow = async (estimitor: KMeans, xData: Tensor | RecursiveArray<number>, params: KElbowPrams = {}): Promise<InertiaItem[]> => {
  const xTensor = checkArray(xData);
  const { k = [ 2, 10 ], verbose = false } = params;
  const [ start, end ] = k;
  const elbowsData = [];
  const {
    tol,
    nInit,
    maxIterTimes,
    init
  } = estimitor;
  const modelVerbosity = estimitor.verbose;
  for (let i = start; i < end; i++) {
    const curEstimator = new KMeans({ tol, nInit, maxIterTimes, init, verbose: modelVerbosity });
    curEstimator.nClusters = i;
    if (verbose) {
      console.log('Start fitting for nCluster=', i);
    }
    await curEstimator.fit(xTensor);
    const inertia = curEstimator.inertiaDense(xTensor, curEstimator.centroids);
    if (verbose) {
      console.log(`nCluster=${i} fitted, inertia=${inertia}`);
    }
    elbowsData.push({
      inertia,
      k: i,
      model: curEstimator
    });
  }
  if (!(xData instanceof Tensor)) {
    dispose(xTensor);
  }
  return elbowsData;
};
