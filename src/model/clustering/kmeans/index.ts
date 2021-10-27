import { BaseClustering, FeatureInputType } from "../../base";
import { tensorEqual } from "../../../linalg";
import { Tensor, sub, norm, gather, stack, min, sum, slice, add, tensor, mul,
  transpose, mean, booleanMaskAsync, equal, argMin, divNoNan, squeeze, zeros, reshape } from "@tensorflow/tfjs-core";

export type KMeansInitType = 'kmeans++' | 'random';
const defaultMaxIterTimes = 10000;
const defaultTol = 1e-5;
export type KMeansParams = {
  nClusters: number,
  tol?: number,
  maxIterTimes?: number,
  randomState?: number
}

export class KMeans extends BaseClustering {
  public nClusters: number;
  public init: KMeansInitType;
  public maxIterTimes: number;
  public tol: number;
  public randomState: number;
  public centroids: Tensor;
  public clusIndex: Tensor;
  private clusCount: Tensor;
  private firstTrainOnBatch: boolean;

  constructor(params: KMeansParams) {
    super();
    if (!params.nClusters) {
      throw new TypeError('nClusters is not specified');
    }
    this.nClusters = params.nClusters;
    this.maxIterTimes = params.maxIterTimes ? params.maxIterTimes : defaultMaxIterTimes;
    this.tol = params.tol ? params.tol : defaultTol;
    this.firstTrainOnBatch = true;
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

  public getClusDist(xTensor: Tensor, centroids: Tensor): Tensor {
    const axisH = 1;
    const newDistsData: Tensor[] = [];
    const nCluster = centroids.shape[0];
    for (let i = 0; i < nCluster; i++) {
      newDistsData.push(norm(sub(xTensor, gather(centroids, i)), 'euclidean', axisH));
    }
    return transpose(stack(newDistsData));
  }

  public getClusIndex(xTensor: Tensor): Tensor {
    const newDists = this.getClusDist(xTensor, this.centroids);
    return argMin(newDists, 1);
  }

  // public async updateChunk(xTensor: Tensor, weights: Tensor): Promise<{ newCentroids: Tensor, newClusIndex: Tensor, weights: number }> {
  //   const axisH = 1;
  //   //const dists = this.getClusDist(xTensor, this.centroids);
  //   const clusIndex = this.getClusIndex(xTensor);
  //   const { newCentroids, newClusIndex } = this.update(xTensor);
  //   //const = this.getClusIndex(xTensor);
  // }

  public async update(xTensor: Tensor): Promise<{ newCentroids: Tensor, newClusIndex: Tensor }> {
    const axisV = 0;
    const newCentroidsData: Tensor[] = [];
    const newClusIndex = this.getClusIndex(xTensor);
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
    this.centroids = this.kmeansPlusPlus(xTensor, this.nClusters);
    for (let i = 0; i < this.maxIterTimes; i++) {
      const { newCentroids, newClusIndex } = await this.update(xTensor);
      if (tensorEqual(newCentroids, this.centroids, this.tol)) {
        return;
      }
      this.centroids = newCentroids;
      this.clusIndex = newClusIndex;
    }
  }

  public async trainOnBatch(xData: FeatureInputType): Promise<void> {
    const xTensor = this.validateData(xData, this.firstTrainOnBatch);
    if (this.firstTrainOnBatch) {
      this.centroids = this.kmeansPlusPlus(xTensor, this.nClusters);
      this.firstTrainOnBatch = false;
      this.clusCount = zeros([ this.nClusters ]);
    }
    const axisV = 0;
    const batchCentroidsData: Tensor[] = [];
    const newClusIndex = this.getClusIndex(xTensor);
    const clusCount: number[] = [];
    for (let i = 0; i < this.nClusters; i++) {
      const mask = equal(newClusIndex, i);
      clusCount.push(sum(mask).dataSync()[0]);
      const centroidI = await booleanMaskAsync(xTensor, mask);
      batchCentroidsData.push(mean(centroidI, axisV));
    }
    // update cluster count
    this.clusCount = add(this.clusCount, tensor(clusCount));
    const batchCentroids = stack(batchCentroidsData);
    //const eta = reshape(divNoNan(1, this.clusCount), [ -1, 1 ]);
    const eta = 0.5;
    // c = (1 - eta) * c + eta * x
    this.centroids = add(mul(this.centroids, sub(1, eta)), mul(batchCentroids, eta));
  }

  public async predict(xData: FeatureInputType): Promise<Tensor> {
    const xTensor = this.validateData(xData, false);
    return this.getClusIndex(xTensor);
  }
}
