import { Criterion } from './criterion';
abstract class Splitter {
  public criterion: Criterion;
  public maxFeatures: number;
  public minSamplesLeaf: number;
  public minWeightLeaf: number;
  public impurityLeft: number;
  public impurityRight: number;
  public weightedNSamples: number;
  public samples: number[];
  // save feature index
  public features: number[];
  // number of feaures
  public nFeaures: number;
  // number of constant features
  public nConstantFeatures: number;
  public sampleWeights: number[];
  public pos: number;
  public start: number;
  public end: number;
  public x: number[][];
  public y: number[];
  
  constructor(criterion: Criterion, maxFeatures: number, minSamplesLeaf: number, minWeightLeaf: number) {
    this.criterion = criterion;
    this.maxFeatures = maxFeatures;
    this.minSamplesLeaf = minSamplesLeaf;
    this.minWeightLeaf = minWeightLeaf;
    this.samples = [];
  }
  private checkInput(xData: number[][], yData: number[], weights: number[]) {
    if (xData instanceof Array) {

    }
    // todo
    return false;
  }
  /**
   * Initialize splitter
   * @param xData feature data
   * @param yData target
   * @param weights weights of samples
   * @returns 
   */
  public init(xData: number[][], yData: number[], weights?: number[]): Splitter {
    const nData = xData.length;
    this.samples = [];
    this.weightedNSamples = 0;
    for (let i = 0; i < nData; i++) {
      if (!weights || weights[i] > 0) {
        this.samples.push(i)
      }
      if (weights) {
        this.weightedNSamples += weights[i]
      } else {
        this.weightedNSamples += weights[i]
      }
    }
    this.x = xData;
    this.y = yData;
    this.sampleWeights = weights;

    return this;
  }

  // public nodeReset(start: number, end: number) {

  // }

  /**
   * Find best split on node samples[start:end]
   * returns -1 in case of failure to allocate memory or 0 otherwise
   */
  public nodeSplit(split: number, nConstantFeaures: number[]) {
    let fi = this.nFeaures;
    let fj: number;
    let nVisitedFeatures = 0;
    let nDrawnConstants = 0;
    // Number of features discovered to be constant during the split search
    let nFoundConstants = 0;
    let nKnownConstants = 0;
    // n_total_constants = n_known_constants + n_found_constants
    let nTotalConstants = nKnownConstants;
    while (fi > this.nConstantFeatures && nVisitedFeatures) {
      fj = nDrawnConstants + Math.floor(Math.random()) * (this.nFeaures);
      if (fj < nKnownConstants) {
        
      } else {
        
      }
    }
  }

  /**
   * reset splitter on node samples[start:end]
   * @param start 
   * @param end 
   * @param weightedNNodeSamples 
   * @returns 
   */
  public nodeReset(start: number, end: number): Splitter {
    this.start = start;
    this.end = end;
    this.criterion.init(this.y, this.samples, this.weightedNSamples, this.start, this.end);
    return this;
  }
}

export class BaseDenseSplitter extends Splitter {
  constructor(criterion: Criterion, maxFeatures: number, minSamplesLeaf: number, minWeightLeaf: number) {
    super(criterion, maxFeatures, minSamplesLeaf, minWeightLeaf);
  }
}

/**
 * Splitter for finding best split
 */
export class BestSplitter extends BaseDenseSplitter {
  /**
   * Find best split on node samples[start:end]
   */
  public nodeSplit() {
    
  }
}
