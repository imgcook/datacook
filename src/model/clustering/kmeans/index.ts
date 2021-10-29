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
const defaultVerbose = false;

/**
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
 *
 *  `verbose` : boolean, default=false\
 *     Verbosity mode.
 */
export type KMeansParams = {
  nClusters?: number,
  tol?: number,
  nInit?: number,
  maxIterTimes?: number,
  init?: KMeansInitType,
  verbose?: boolean,
};

export class KMeans extends BaseClustering {
  public nClusters: number;
  public init: KMeansInitType;
  public nInit: number;
  public maxIterTimes: number;
  public tol: number;
  public randomState: number;
  public verbose: boolean;
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
   *
   *  `verbose` : boolean, default=false\
   *     Verbosity mode.
   *
   * Examples
   * --------
   *
   * ### Basic Usage
   *
   * ``` javascript
   * import * as Datacook from 'datacook';
   * const { KMeans } = DataCook.Model;
   * const xData = [
   *  [1, 2], [1, 4], [1, 0],
   *  [10, 2], [10, 4], [10, 0]
   * ];
   * const kmeans = new KMeans({ nClusters: 3 });
   * await kmeans.fit(clusData);
   * const predClus = await kmeans.predict(xData);
   * predClus.print();
   * // Tensor
   * // [0, 0, 0, 1, 1, 1]
   *
   * // save and load model
   * const modelJSON = await kmeans.toJson();
   * const kmeans2 = new KMeans({});
   * kmeans2.fromJson(modelJSON);
   * const predClus = await kmeans2.predict(xData);
   * predClus.print();
   * // Tensor
   * // [0, 0, 0, 1, 1, 1]
   * ```
   *
   * ### Train on batch
   * ```javascript
   * import * as Datacook from 'datacook';
   * import * as tf from '@tensorflow/tfjs-core';
   * const { KMeans } = DataCook.Model;
   *
   * // create dataset
   * const clust1 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 5, 5 ]));
   * const clust2 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ 10, 0 ]));
   * const clust3 = tf.add(tf.mul(tf.randomNormal([ 100, 2 ]), tf.tensor([ 2, 2 ])), tf.tensor([ -10, 0 ]));
   * const clusData = tf.concat([ clust1, clust2, clust3 ]);
   * // fit kmeans model
   * const kmeans = new KMeans({ nClusters: 3 });
   * const batchSize = 30;
   * const epochSize = Math.floor(clusData.shape[0] / batchSize);
   * for (let i = 0; i < 50; i++) {
   *    const j = Math.floor(i % epochSize);
   *    const batchX = tf.slice(clusData, [j * batchSize, 0], [batchSize ,2]);
   *    await kmeans.trainOnBatch(batchX);
   * }
   * const predClus = await kmeans.predict(clusData);
   * const accuracy = await checkClusAccuracy(predClus);
   * console.log('accuracy:', accuracy);
   * // accuracy: 0.9666666666666667
   * ```
   **/
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
    this.verbose = params.verbose ? !!params.verbose : defaultVerbose;
  }

  /**
   * Get a set of centroids according to input tensor and type of initialize
   * @param xTensor tensor of input data
   * @returns tensor of centroids
   */
  public getInitCentroids(xTensor: Tensor): Tensor {
    if (this.init === 'kmeans++') {
      return this.kmeansPlusPlus(xTensor, this.nClusters);
    }
    if (this.init === 'random') {
      const shuffleIndices = Array.from(Array(xTensor.shape[0]).keys());
      shuffle(shuffleIndices);
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
   * @param xTensor tensor of input data
   * @returns centroids seledted from nInit iterations
   */
  public async initCentroids(xTensor: Tensor): Promise<{ selectedCentroids: Tensor, inertia: number }> {
    // If initialized centroids are defined
    if (this.init instanceof Tensor || this.init instanceof Array) {
      const selectedCentroids = this.getInitCentroids(xTensor);
      const inertia = this.inertiaDense(xTensor, selectedCentroids);
      return { selectedCentroids, inertia };
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
    return { selectedCentroids, inertia: minInertia };
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
   * @param xTensor tensor of x data
   * @param centroids tensor of centroids
   */
  public validataCentroidsShape(xTensor: Tensor, centroids: Tensor): void {
    if (centroids.shape[0] !== this.nClusters) {
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
   * @param xTensor tensor of x data
   * @param centroids tensor of centroids
   * @returns tensor of distance from input samples and its assigned cluster center
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
   * @param xTensor tensor of input data
   * @returns tensor of assigned cluster index
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
   * @param xData input data of shape (nSamples, nFeatures) in type of array or tensor
   */
  public async fit(xData: FeatureInputType): Promise<void> {
    const xTensor = this.validateData(xData, true);
    let oldClusIndex: Tensor;
    const { selectedCentroids, inertia } = await this.initCentroids(xTensor);
    this.centroids = selectedCentroids;
    if (this.verbose) {
      console.log(`Initialization complete with inertia ${inertia}`);
    }
    for (let i = 0; i < this.maxIterTimes; i++) {
      const { newCentroids, newClusIndex } = await this.update(xTensor, this.centroids);

      if (this.verbose) {
        const inertia = this.inertiaDense(xTensor, this.centroids);
        console.log(`Iteration ${i}, inertia ${inertia}.`);
      }

      // check strict convergence
      if (oldClusIndex && tensorEqual(newClusIndex, oldClusIndex)) {
        if (this.verbose) {
          console.log(`Converged at iteration ${i}: strict converge.`);
        }
        return;
      }

      // No strict convergence, check for tol based convergence.
      if (tensorEqual(newCentroids, this.centroids, this.tol)) {
        if (this.verbose) {
          console.log(`Converged at iteration ${i}: center shift within tolerance ${this.tol}.`);
        }
        return;
      }
      this.centroids = newCentroids;
      oldClusIndex = newClusIndex;
    }
  }

  /**
   * Train kmeans model by batch. Here we apply mini-batch kmeans algorithm to
   * update centroids in each iteration.
   * @param xData input data of shape (batchSize, nFeatures) in type of array or tensor
   */
  public async trainOnBatch(xData: FeatureInputType): Promise<void> {
    const xTensor = this.validateData(xData, this.firstTrainOnBatch);
    if (this.firstTrainOnBatch) {
      this.centroids = (await this.initCentroids(xTensor)).selectedCentroids;
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
   * @param xData input data of shape (nSamples, nFeatures) in type of array or tensor
   * @returns tensor of redicted cluster index
   */
  public async predict(xData: FeatureInputType): Promise<Tensor> {
    const xTensor = this.validateData(xData, false);
    return this.getClusIndex(xTensor, this.centroids);
  }

  /**
   * Get scores for input xData on the kmeans model.\
   * score = -inertia, larger score usually represent better fit.
   * @param xData input data
   * @returns tensor of -inertia
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
    this.nFeature = params.nFeature;
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
      centroids: await this.centroids.array(),
      nFeature: this.nFeature
    };
    return JSON.stringify(modelParams);
  }
}
