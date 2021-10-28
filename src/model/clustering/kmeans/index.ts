import { BaseClustering, FeatureInputType } from '../../base';
import { tensorEqual } from '../../../linalg';
import { Tensor, sub, norm, gather, stack, min, sum, slice, add, tensor, mul,
  transpose, mean, booleanMaskAsync, equal, argMin, divNoNan, squeeze, zeros, reshape, RecursiveArray, square, neg } from '@tensorflow/tfjs-core';
import { shuffle } from '../../../generic';
import { checkArray } from '../../../utils/validation';

export type KMeansInitType = 'kmeans++' | 'random' | Tensor | RecursiveArray<number>;
const defaultMaxIterTimes = 1000;
const defaultNClusters = 8;
const defaultTol = 1e-5;
const defaultNInit = 10;
const defaultInit = 'kmeans++';

export type KMeansParams = {
  nClusters?: number,
  tol?: number,
  nInit?: number,
  maxIterTimes?: number,
  init?: KMeansInitType,
};

export class KMeans extends BaseClustering {
  public nClusters: number;
  public init: KMeansInitType;
  public nInit: number;
  public maxIterTimes: number;
  public tol: number;
  public randomState: number;
  public centroids: Tensor;
  private clusWeightedSum: Tensor;
  private firstTrainOnBatch: boolean;

  /**
   * Initialize KMeans model.
   * @param params parameter object used to initialize KMeans model.
   *
   * Options in params
   * ------------
   *
   * `nClusters` : number, default=8\
   *    The number of clusters to form as well as the number of centroids
   *    to generate.
   *
   * `init`: {'kmeans++', 'random'} or an array of shape (nClusters, nFeatures),
   *    default = 'kmeans++'
   *
   *    - 'kmeans++': select initial cluster centroids using kmeans++
   *    - 'random': randomly select initial centroids
   *    - array of shape (nClusters, nFeatures): use user defined centroids.
   *
   * `nInit`: number, default=10.\
   *    Number of time the algorithm will be run with different initialization
   *    centroid seeds. The final results will be the best output of nInit
   *    consecutive runs in terms of inertia.
   *
   *  `maxIterTimes` : number, default=1000\
   *    Maximum number of iterations of the k-means algorithm for a single run.
   *
   *  `tol` : number, default=1e-5\
   *    Relative tolerance with regards to Frobenius norm of the difference
   *    in the cluster centers of two consecutive iterations to declare
   *    convergence.
   */
  constructor(params: KMeansParams) {
    super();
    if (!params.nClusters) {
      throw new TypeError('nClusters is not specified');
    }
    this.nInit = params.nInit ? params.nInit : defaultNInit;
    this.nClusters = params.nClusters ? params.nClusters : defaultNClusters;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : defaultMaxIterTimes;
    this.tol = params.tol ? params.tol : defaultTol;
    this.firstTrainOnBatch = true;
    this.init = params.init ? params.init : defaultInit;
  }

  /**
   * Get a set of centroids according to input tensor and type of initialize
   * @param xTensor
   * @returns
   */
  public getInitCentroids(xTensor: Tensor): Tensor {
    if (this.init === 'kmeans++') {
      return this.kmeansPlusPlus(xTensor, this.nClusters);
    }
    if (this.init === 'random') {
      const shuffleIndices = Array.from(Array(xTensor.shape[0]).keys());
      shuffle(Array.from(Array(xTensor.shape[0]).keys()));
      return gather(xTensor, shuffleIndices.slice(this.nClusters));
    }
    if (this.init instanceof Tensor || this.init instanceof Array) {
      const initTensor = checkArray(this.init, 'float32', 2);
      this.validataCentroidsShape(xTensor, initTensor);
      return initTensor;
    }
    return this.kmeansPlusPlus(xTensor, this.nClusters);
  }

  /**
   * Initialize centroids
   * @param xTensor
   * @returns centroids seledted from nInit iterations
   */
  public async initCentroids(xTensor: Tensor): Promise<Tensor> {
    // If initialized centroids are defined
    if (this.init instanceof Tensor || this.init instanceof Array) {
      return this.getInitCentroids(xTensor);
    }
    // Initialize for nInit times to find the best centroids according to interia criteria.
    let minInertia = Number.MAX_SAFE_INTEGER;
    let selectedCentroids: Tensor;
    for (let i = 0; i < this.nInit; i++) {
      const centroids = this.getInitCentroids(xTensor);
      const { newCentroids } = await this.update(xTensor, centroids);
      const inertia = this.inertiaDense(xTensor, newCentroids);
      if (inertia < minInertia) {
        minInertia = inertia;
        selectedCentroids = newCentroids;
      }
    }
    return selectedCentroids;
  }

  /**
   * Compute inertia for dense input data and given centroids.
   * Sum of squared distance between each sample and its assigned center.
   * @param xTensor
   * @param centroids
   * @returns computed inertia
   */
  public inertiaDense(xTensor: Tensor, centroids: Tensor): number {
    const axisH = 1;
    const dists = this.getClusDist(xTensor, centroids);
    const minDists = min(dists, axisH);
    return sum(square(minDists)).dataSync()[0];
  }

  /**
   * Apply kmeans plus plus algorithm to initialize centroids.
   * @param xTensor Tensor of input data.
   * @param k Number of centroids to initialize.
   */
  public kmeansPlusPlus(xTensor: Tensor, k: number): Tensor {
    const centroids: Tensor[] = [];
    const nData = xTensor.shape[0];
    const axisH = 1;
    const initRnd = Math.floor(Math.random() * nData);

    centroids.push(squeeze(gather(xTensor, [ initRnd ])));
    while (centroids.length < k) {
      const dists = this.getClusDist(xTensor, stack(centroids));
      const minDists = min(dists, axisH);
      const sumDists = sum(minDists);
      const probs = divNoNan(minDists, sumDists);
      const rnd = Math.random();
      let cumProb = 0;
      let idx = 0;
      for (; idx < nData - 1; idx++) {
        cumProb += slice(probs, idx, 1).dataSync()[0];
        if (cumProb > rnd) {
          break;
        }
      }
      centroids.push(squeeze(gather(xTensor, [ idx ])));
    }
    return stack(centroids);
  }

  /**
   * Check if centers is compatible with X and nClusters
   * @param xTensor
   * @param centroids
   */
  public validataCentroidsShape(xTensor: Tensor, centroids: Tensor): void {
    if (centroids.shape[1] !== this.nClusters) {
      throw new TypeError(
        `The shape of the initial centers ${centroids.shape} does not match the number of clusters ${this.nClusters}.`
      );
    }
    if (centroids.shape[1] !== xTensor.shape[1]) {
      throw new TypeError(
        `The shape of the initial centers ${centroids.shape[1]} does not match the number of features of the data ${xTensor.shape[1]}.`
      );
    }
  }

  /**
   * Get distance from data samples to their assigned cluster centers.
   * @param xTensor
   * @param centroids
   * @returns
   */
  public getClusDist(xTensor: Tensor, centroids: Tensor): Tensor {
    const axisH = 1;
    const newDistsData: Tensor[] = [];
    const nCluster = centroids.shape[0];
    for (let i = 0; i < nCluster; i++) {
      newDistsData.push(norm(sub(xTensor, gather(centroids, i)), 'euclidean', axisH));
    }
    return transpose(stack(newDistsData));
  }


  /**
   * Get the cluster index of given samples
   * @param xTensor
   * @returns
   */
  public getClusIndex(xTensor: Tensor, centroids: Tensor): Tensor {
    const newDists = this.getClusDist(xTensor, centroids);
    return argMin(newDists, 1);
  }

  /**
   * Update centroids in one step.
   * @param xTensor input tensor
   * @param centroids centroids
   * @returns `{ newCentroids, newClusIndex }`
   */
  public async update(xTensor: Tensor, centroids: Tensor): Promise<{ newCentroids: Tensor, newClusIndex: Tensor }> {
    const axisV = 0;
    const newCentroidsData: Tensor[] = [];
    const newClusIndex = this.getClusIndex(xTensor, centroids);
    for (let i = 0; i < this.nClusters; i++) {
      const mask = equal(newClusIndex, i);
      const centroidI = await booleanMaskAsync(xTensor, mask);
      newCentroidsData.push(mean(centroidI, axisV));
    }
    const newCentroids = stack(newCentroidsData);
    return { newCentroids, newClusIndex };
  }

  /**
   * Fit kmeans model
   * @param xData
   * @returns
   */
  public async fit(xData: FeatureInputType): Promise<void> {
    const xTensor = this.validateData(xData, true);
    this.centroids = await this.initCentroids(xTensor);
    for (let i = 0; i < this.maxIterTimes; i++) {
      const { newCentroids } = await this.update(xTensor, this.centroids);
      if (tensorEqual(newCentroids, this.centroids, this.tol)) {
        return;
      }
      this.centroids = newCentroids;
    }
  }

  /**
   * Train kmeans model by batch. Here we apply mini-batch kmeans algorithm to
   * update centroids in each iteration.
   * @param xData
   */
  public async trainOnBatch(xData: FeatureInputType): Promise<void> {
    const xTensor = this.validateData(xData, this.firstTrainOnBatch);
    if (this.firstTrainOnBatch) {
      this.centroids = await this.initCentroids(xTensor);
      this.firstTrainOnBatch = false;
      this.clusWeightedSum = zeros([ this.nClusters ]);
    }
    const axisV = 0;
    const batchCentroidsData: Tensor[] = [];
    const newClusIndex = this.getClusIndex(xTensor, this.centroids);
    const clusCount: number[] = [];
    for (let i = 0; i < this.nClusters; i++) {
      const mask = equal(newClusIndex, i);
      clusCount.push(sum(mask).dataSync()[0]);
      const centroidI = await booleanMaskAsync(xTensor, mask);
      batchCentroidsData.push(mean(centroidI, axisV));
    }
    const batchCentroids = stack(batchCentroidsData);
    const newClusWeightedSum = add(this.clusWeightedSum, tensor(clusCount));
    // update centroids
    this.centroids = add(mul(this.centroids, reshape(this.clusWeightedSum, [ -1, 1 ])), mul(batchCentroids, reshape(clusCount, [ -1, 1 ])));
    this.centroids = divNoNan(this.centroids, reshape(newClusWeightedSum, [ -1, 1 ]));
    // update cluster count
    this.clusWeightedSum = newClusWeightedSum;
  }

  /**
   * Predict sample clusters for given input.
   * @param xData
   * @returns predicted cluster index
   */
  public async predict(xData: FeatureInputType): Promise<Tensor> {
    const xTensor = this.validateData(xData, false);
    return this.getClusIndex(xTensor, this.centroids);
  }

  /**
   * Get scores for input xData on the kmeans model.\
   * score = -inertia, larger score usually represent better fit.
   * @param xData
   * @returns
   */
  public async score(xData: FeatureInputType): Promise<number> {
    const xTensor = this.validateData(xData, false);
    const interia = this.inertiaDense(xTensor, this.centroids);
    return neg(interia).dataSync()[0];
  }

  /**
   * Load model paramters from json string object
   * @param modelJson model json saved as string object
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<KMeans> {
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
    this.firstTrainOnBatch = true;
    this.init = params.init ? params.init : defaultInit;
    this.centroids = params.centroids ? tensor(params.centroids) : null;
    return this;
  }

  /**
   * Dump model parameters to json string.
   * @returns Stringfied model parameters
   */
  public async toJson(): Promise<string> {
    const modelParams = {
      name: 'KMeans',
      nInit: this.nInit,
      nClusters: this.nClusters,
      maxIterTimes: this.maxIterTimes,
      tol: this.tol,
      firstTrainOnBatch: this.firstTrainOnBatch,
      init: this.init,
      centroids: await this.centroids.array()
    };
    return JSON.stringify(modelParams);
  }
}
