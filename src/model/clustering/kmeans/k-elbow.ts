import { dispose, RecursiveArray, Tensor } from '@tensorflow/tfjs-core';
import { KMeans } from '.';
import { checkArray } from '../../../utils/validation';
import { BaseClustering } from '../../base';

// export class kElbow
export interface InertiaItem {
  inertia: number;
  k: number;
}
export interface KElbowPrams {
  k?: [ number, number ],
  verbose?: boolean
}

export class KElbow {
  public models: KMeans[];
  constructor() {
       
  }
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
  for (let i = start; i < end; i++) {
    estimitor.nClusters = i;
    if (verbose) {
      console.log('Start fitting for nCluster=', i);
    }
    await estimitor.fit(xTensor);
    const inertia = estimitor.inertiaDense(xTensor, estimitor.centroids);
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
