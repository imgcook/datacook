import { BaseClustring } from "../base";
import { Matrix } from 'ml-matrix';

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
  public centroids: number[][];
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
    this.centroids = params.centroids ? params.centroids : null;
    this.nFeature = params.nFeature;
  }


  /**
   * Get distance from data samples to their assigned cluster centers.
   * @param xTensor tensor of x data
   * @param centroids tensor of centroids
   * @returns tensor of distance from input samples and its assigned cluster center
   */
  public getClusDist(x: number[][], centroids: number[][]): number[][] {
    const newDistsData: number[][] = [];
    const xMatrix = new Matrix(x);
    const nCluster = centroids.length;
    for (let i = 0; i < nCluster; i++) {
      newDistsData.push(xMatrix.subColumnVector(centroids[i]).pow(2).sum('row'));
    }
    return newDistsData;
  }


  /**
   * Get the cluster index of given samples
   * @param xTensor tensor of input data
   * @returns tensor of assigned cluster index
   */
  public getClusIndex(xTensor: number[][], centroids: number[][]): number[] {
    const newDists = this.getClusDist(xTensor, centroids);
    return (new Matrix(newDists)).minColumnIndex(0);
  }
  /**
   * Predict sample clusters for given input.
   * @param xData input data of shape (nSamples, nFeatures) in type of array or tensor
   * @returns tensor of redicted cluster index
   */
  public async predict(xData: number[][]): Promise<number[]> {
    this.validateData(xData);
    return this.getClusIndex(xData, this.centroids);
  }
}
