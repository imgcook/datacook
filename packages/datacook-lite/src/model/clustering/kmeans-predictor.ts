import { BaseClustring } from "../base";
import { matrix, createZeroMatrix, Vector } from "../../core/classes";
import { sub2d, argMin2d, sum2d, square2d } from "../../core/op";
import { Matrix } from "../../backend-cpu/classes";

export type KMeansInitType = 'kmeans++' | 'random';
const defaultMaxIterTimes = 1000;
const defaultTol = 1e-5;
const defaultNInit = 10;
const defaultInit = 'kmeans++';


export class KMeansPredictor extends BaseClustring {
  public nClusters: number;
  public init: KMeansInitType;
  public nInit: number;
  public maxIterTimes: number;
  public tol: number;
  public randomState: number;
  public verbose: boolean;
  public centroids: Matrix;
  /**
   * Load model parameters from json string object
   * @param modelJson model json saved as string object
   */
  public async fromJson(modelJson: string): Promise<void> {
    const params = JSON.parse(modelJson);
    if (params.name !== 'KMeans') {
      throw new TypeError(`${params.name} is not KMeans`);
    }
    if (!params.nClusters) {
      throw new TypeError('nClusters is not specified');
    }
    this.nInit = params.nInit ? params.nInit : defaultNInit;
    this.nClusters = params.nClusters;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : defaultMaxIterTimes;
    this.tol = params.tol ? params.tol : defaultTol;
    this.init = params.init ? params.init : defaultInit;
    this.centroids = params.centroids ? matrix(params.centroids) : null;
    this.nFeature = params.nFeature;
  }


  /**
   * Get distance from data samples to their assigned cluster centers.
   * @param xTensor tensor of x data
   * @param centroids tensor of centroids
   * @returns tensor of distance from input samples and its assigned cluster center
   */
  public getClusDist(x: Matrix, centroids: Matrix): Matrix {
    const xMatrix = x;
    const centroidsMatrix = centroids;
    const [ n, m ] = xMatrix.shape;
    const distsMatrix = createZeroMatrix(n, m);
    const nCluster = centroids.shape[0];
    for (let i = 0; i < nCluster; i++) {
      const dist = sum2d(square2d(sub2d(xMatrix, centroidsMatrix.getRow(i), 0)), 1) as Vector;
      distsMatrix.setColumn(i, dist);
    }
    return distsMatrix;
  }


  /**
   * Get the cluster index of given samples
   * @param xTensor tensor of input data
   * @returns tensor of assigned cluster index
   */
  public getClusIndex(x: Matrix, centroids: Matrix): Vector {
    const newDists = this.getClusDist(x, centroids);
    return (argMin2d(newDists, 1) as Vector);
  }
  /**
   * Predict sample clusters for given input.
   * @param xData input data of shape (nSamples, nFeatures) in type of array or tensor
   * @returns tensor of redicted cluster index
   */
  public async predict(xData: number[][]): Promise<number[]> {
    // console.time('validate');
    this.validateData(xData);
    // console.timeEnd('validate');
    // console.time('create');
    const x = matrix(xData);
    // console.timeEnd('create');
    return this.getClusIndex(x, this.centroids).data;
  }
}
