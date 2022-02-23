import { Criterion } from './criterion';
import { sort } from './utils';

export const FEATURE_THRESHOLD = 1e-7;

export interface SplitRecord {
  impurityLeft: number;
  impurityRight: number;
  pos: number;
  feature: number;
  threshold: number;
  improvement: number;
}

export abstract class Splitter {
  public criterion: Criterion;
  public nSamples: number;
  public samples: number[];
  public nFeatures: number;
  public featureValues: number[][];
  public maxFeatures: number;
  public features: number[];
  public minSampleLeaf: number;
  public minWeightLeaf: number;
  public impurityLeft: number;
  public impurityRight: number;
  public threshold: number;
  public improvement: number;
  public sampleWeight: number[];
  public weightedNSamples: number;
  public X: number[][];
  public y: number[];
  public pos: number;
  public start: number;
  public end: number;

  constructor(criterion: Criterion, maxFeatures: number, minSampleLeaf: number, minWeightLeaf: number) {
    this.criterion = criterion;
    this.nSamples = 0;
    this.nFeatures = 0;
    this.maxFeatures = maxFeatures;
    this.minSampleLeaf = minSampleLeaf;
    this.minWeightLeaf = minWeightLeaf;
  }

  /**
   * Init spliiter
   * Take in the input data X, the target Y, and optional sample weights.
   * @param X input features
   * @param y targets vector
   * @param sampleWeight sample weights
   */
  public init(X: number[][], y: number[], sampleWeight?: number[]): void {
    this.nSamples = X.length;
    this.samples = [];
    // let j = 0;
    for (let i = 0; i < this.nSamples; i++) {
      if (sampleWeight && sampleWeight[i] != 0 || !sampleWeight) {
        this.samples.push(i);
        // j += 1;
      }
      if (sampleWeight) {
        this.weightedNSamples += sampleWeight[i];
      } else {
        this.weightedNSamples += 1.0;
      }
    }
    this.nFeatures = X[0].length;
    this.featureValues = X;
    this.sampleWeight = sampleWeight;
    this.features = Array.from(new Array(this.nFeatures).keys());
    this.y = y;
    this.X = X;
  }

  public nodeReset(start: number, end: number): void {
    this.start = start;
    this.end = end;
    this.criterion.init(this.y, this.sampleWeight, this.samples, this.weightedNSamples, this.start, this.end);
  }

  public nodeImpurity(): number {
    return this.criterion.nodeImpurity();
  }

  abstract nodeSplit(impurity: number, nConstantFeatures: number): {split: SplitRecord, nConstantFeatures: number};
}

// export class BaseDenseSplitter extends Splitter {
//   public init(X: number[][], y: number[], sampleWeight: number[]) {
//     this.init(X, y, sampleWeight);
//   }
// }

export class BestSplitter extends Splitter {
  public nodeSplit(impurity: number, nConstantFeatures: number): {split: SplitRecord, nConstantFeatures: number} {
    const samples = this.samples;
    const start = this.start;
    const end = this.end;
    let nVisitedFeatures = 0;
    let nFoundConstants = 0;
    let nDrawnConstants = 0;
    const nKnownConstants = nConstantFeatures;
    let nTotalConstants = nKnownConstants;
    let currentFeature: number;
    let featureX: number[];
    let bestProxyImprovement = Number.MIN_SAFE_INTEGER;

    let fi = this.nFeatures;
    let fj;

    let bestThreshold: number;
    const bestSplit: SplitRecord = {
      pos: end,
      feature: -1,
      threshold: Number.MIN_SAFE_INTEGER,
      improvement: Number.MIN_SAFE_INTEGER,
      impurityLeft: Number.MAX_SAFE_INTEGER,
      impurityRight: Number.MAX_SAFE_INTEGER
    };

    while (fi > nTotalConstants && (nVisitedFeatures < this.maxFeatures ||
      nVisitedFeatures <= nFoundConstants + nDrawnConstants)) {

      nVisitedFeatures += 1;
      fj = nDrawnConstants + Math.floor(Math.random() * (fi - nFoundConstants));

      if (fj < nKnownConstants) {
        [ this.features[nDrawnConstants], this.features[fj] ] = [ this.features[fj], this.features[nDrawnConstants] ];
        nDrawnConstants += 1;
      } else {
        fj += nFoundConstants;
        currentFeature = this.features[fj];
        featureX = new Array(this.nSamples);
        for (let i = this.start; i < this.end; i++) {
          const p = samples[i];
          featureX[i] = this.X[p][currentFeature];
        }

        sort(featureX, samples, start, end);

        if (featureX[end - 1] <= featureX[start] + FEATURE_THRESHOLD) {
          [ this.features[fj], this.features[nTotalConstants] ] = [ this.features[nTotalConstants], this.features[fj] ];
          nFoundConstants += 1;
          nTotalConstants += 1;
        } else {
          fi -= 1;
          [ this.features[fi], this.features[fj] ] = [ this.features[fj], this.features[fi] ];
          this.criterion.reset();
          let p = start;
          while (p < end) {
            while (p + 1 < end && Math.abs(featureX[p + 1] - featureX[p]) <= FEATURE_THRESHOLD) {
              p += 1;
            }
            p += 1;
            if (p < end) {
              const currentPos = p;
              if ((currentPos - start) < this.minSampleLeaf || (end - currentPos) < this.minSampleLeaf) {
                continue;
              }
              this.criterion.update(samples, currentPos);

              if ((this.criterion.weightedNLeft < this.minWeightLeaf) ||
                this.criterion.weightedNRight < this.minWeightLeaf) {
                continue;
              }
              const currentProxyImprovement = this.criterion.proxyImpurityImprovement();
              if (currentProxyImprovement > bestProxyImprovement) {
                // sum of halves is used to avoid infinite value
                let currentThreshold = featureX[p - 1] / 2.0 + featureX[p] / 2.0;

                if ((currentThreshold === featureX[p]) ||
                  currentThreshold === Number.MAX_SAFE_INTEGER ||
                  currentThreshold === Number.MIN_SAFE_INTEGER) {

                  currentThreshold = featureX[p - 1];
                }
                bestProxyImprovement = currentProxyImprovement;
                bestSplit.pos = currentPos;
                const { impurityLeft, impurityRight } = this.criterion.childrenImpurity();
                bestSplit.impurityLeft = impurityLeft;
                bestSplit.impurityRight = impurityRight;
                bestSplit.improvement = this.criterion.impurityImprovement(impurity, impurityLeft, impurityRight);
                bestSplit.feature = currentFeature;
                bestSplit.threshold = currentThreshold;
              }
            }
          }
        }
      }

    }
    // Reorganize into samples[start:best.pos] + samples[best.pos:end]
    if (bestSplit.pos < end && bestSplit.pos !== -1) {
      let partitionEnd = end;
      let p = start;
      while (p < partitionEnd) {
        if (this.X[this.samples[p]][bestSplit.feature] <= bestThreshold) {
          p += 1;
        } else {
          partitionEnd -= 1;
          [ this.samples[p], this.samples[partitionEnd] ] = [ this.samples[partitionEnd], this.samples[p] ];
        }
      }
      // this.criterion.reset();
      // this.criterion.update(this.samples, bestSplit.pos);
      // const { impurityLeft, impurityRight } = this.criterion.childrenImpurity();
      // bestSplit.impurityLeft = impurityLeft;
      // bestSplit.impurityRight = impurityRight;
      // bestSplit.improvement = this.criterion.impurityImprovement(impurity, impurityLeft, impurityRight);
    }
    return { split: bestSplit, nConstantFeatures: nTotalConstants };
  }
}
