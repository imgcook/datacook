/**
 * Includes different split criterions for decision tree
 * Code of this part refers to the implementation in scikit learn
 */

export abstract class Criterion {
  public sampleWeight: number[];
  public weightedNLeft: number;
  public weightedNRight: number;
  public weightedNSamples: number;
  public weightedNNodeSamples: number;
  /**
   * The number of target, the dimensionality of the prediction
   */
  public nOutputs: number;
  /**
   * The number of classes in target
   */
  public nClasses: number;
  public pos: number;
  public start: number;
  public end: number;
  public sumLeft: number[];
  public sumRight: number[];
  public sumTotal: number[];
  public samples: number[];
  public y: number[];

  /**
   * calculating the impurity of the node
   */
  abstract nodeImpurity(): number;
  abstract childrenImpurity(): { impurityLeft: number, impurityRight: number };
  abstract update (samples: number[], newPos: number): void;
  abstract reset(): void;
  abstract nodeValue(): number[];
  abstract leftValue(): number[];
  abstract rightValue(): number[];
  /**
   * Initialize the criterion.
   * @param y The target stored as a buffer for memory efficiency
   * @param samples The weight of each sample
   * @param weightedNSamples The total weight of all samples
   * @param start The first sample to use in the mask
   * @param end The last sample to use in the mask
   */
  public init(y: number[], sampleWeight: number[], samples: number[], weightedNSamples: number, start: number, end: number): void {
    this.weightedNSamples = weightedNSamples;
    this.samples = samples;
    this.y = y;
    this.start = start;
    this.end = end;
    this.weightedNNodeSamples = 0;
    this.sampleWeight = sampleWeight;
    this.nClasses = Math.max(...y) + 1;
    this.sumTotal = new Array(this.nClasses).fill(0);
    this.sumLeft = new Array(this.nClasses).fill(0);
    this.sumRight = new Array(this.nClasses).fill(0);
    for (let i = start; i < end; i++) {
      const pos = samples[i];
      let w = 1;
      if (this.sampleWeight && this.sampleWeight instanceof Array && this.sampleWeight.length) {
        w = sampleWeight[pos];
      }
      this.weightedNNodeSamples += w;
      const k = y[pos];
      this.sumTotal[k] += w;
    }
  }
  /**
   * Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
   * @returns proxy improvement
   */
  public proxyImpurityImprovement (): number {
    const { impurityLeft, impurityRight } = this.childrenImpurity();
    return (- this.weightedNLeft * impurityLeft - this.weightedNRight * impurityRight);
  }
  public impurityImprovement (impurityParent: number, impurityLeft: number, impurityRight: number): number {
    return ((this.weightedNNodeSamples / this.weightedNSamples) * (
      impurityParent - (this.weightedNRight / this.weightedNNodeSamples * impurityRight) - (
        this.weightedNLeft / this.weightedNNodeSamples * impurityLeft
      )
    ));
  }
}

export abstract class ClassificationCriterion extends Criterion {
  /**
   * Reset the criterion at pos=start.
   */
  public reset(): void {
    this.weightedNLeft = 0;
    this.weightedNRight = this.weightedNNodeSamples;
    this.pos = this.start;
    for (let k = 0; k < this.nClasses; k++) {
      this.sumLeft[k] = 0;
      this.sumRight[k] = this.sumTotal[k];
    }
  }
  /**
   * Reset the criterion at pos=end.
   */
  public reverseReset(): void {
    this.weightedNRight = 0;
    this.weightedNLeft = this.weightedNNodeSamples;
    this.pos = this.end;
    for (let k = 0; k < this.nClasses; k++) {
      this.sumLeft[k] = this.sumTotal[k];
      this.sumRight[k] = 0;
    }
  }
  /**
   * update statistics by moving samples to left child
   * @param newPos new ending position for which to move samples from right child to left child.
   */
  public update(samples: number[], newPos: number): void {
    if (newPos - this.pos <= this.end - newPos) {
      for (let p = this.pos; p < newPos; p++) {
        const i = samples[p];
        const w = this.sampleWeight ? this.sampleWeight[i] : 1;
        const labelIndex = this.y[i];
        this.sumLeft[labelIndex] += w;
        this.weightedNLeft += w;
      }
    } else {
      this.reverseReset();
      for (let p = this.end - 1; p > newPos - 1; p--) {
        const i = samples[p];
        const w = this.sampleWeight ? this.sampleWeight[i] : 1;
        const labelIndex = this.y[i];
        this.sumLeft[labelIndex] -= w;
        this.weightedNLeft -= w;
      }
    }
    // update right part statistics
    this.weightedNRight = this.weightedNNodeSamples - this.weightedNLeft;
    for (let k = 0; k < this.nClasses; k++) {
      this.sumRight[k] = this.sumTotal[k] - this.sumLeft[k];
    }
    this.pos = newPos;
  }
  /**
   * Compute the node value of samples[start:end] into dest.
   */
  public nodeValue(): number[] {
    return [ ...this.sumTotal ];
  }
  public leftValue(): number[] {
    return [ ...this.sumLeft ];
  }
  public rightValue(): number[] {
    return [ ...this.sumRight ];
  }
}

/**
 * Evaluate the cross-entropy criterion as impurity of current node,
 * i.e. the impurity of samples[start:end]. The smaller the impurity the
 * better.
 */
export class EntropyCriterion extends ClassificationCriterion {
  public nodeImpurity = (): number => {
    let entropy = 0;
    for (let k = 0; k < this.nClasses; k++) {
      const countK = this.sumTotal[k];
      if (countK > 0.0) {
        const pK = countK * 1.0 / this.weightedNNodeSamples;
        entropy -= pK * Math.log(pK);
      }
    }
    return entropy;
  };
  /**
   * Evaluate the impurity in children nodes.
   */
  public childrenImpurity = (): { impurityLeft: number, impurityRight: number } => {
    let entropyLeft = 0;
    let entropyRight = 0;
    for (let k = 0; k < this.nClasses; k++) {
      const countKLeft = this.sumLeft[k];
      const countKRight = this.sumRight[k];
      if (countKLeft > 0.0) {
        const pk = countKLeft / this.weightedNLeft;
        entropyLeft -= pk * Math.log(pk);
      }
      if (countKRight > 0.0) {
        const pk = countKRight / this.weightedNRight;
        entropyRight -= pk * Math.log(pk);
      }
    }
    return {
      impurityLeft: entropyLeft,
      impurityRight: entropyRight
    };
  };
}

/**
 * Gini impurity
 */
export class GiniCriterion extends ClassificationCriterion {
  public nodeImpurity = (): number => {
    let sqCount = 0;
    for (let k = 0; k < this.nClasses; k++) {
      const countK = this.sumTotal[k];
      sqCount += countK * countK;
    }
    const gini = 1.0 - sqCount / (this.weightedNNodeSamples * this.weightedNNodeSamples);
    return gini;
  };

  public childrenImpurity = (): { impurityLeft: number, impurityRight: number } => {
    let sqCountLeft = 0;
    let sqCountRight = 0;
    for (let k = 0; k < this.nClasses; k++) {
      const countKLeft = this.sumLeft[k];
      const countKRight = this.sumRight[k];
      sqCountLeft += countKLeft * countKLeft;
      sqCountRight += countKRight * countKRight;
    }
    const giniLeft = 1.0 - sqCountLeft / (this.weightedNLeft * this.weightedNLeft);
    const giniRight = 1.0 - sqCountRight / (this.weightedNRight * this.weightedNRight);
    return { impurityLeft: giniLeft, impurityRight: giniRight };
  };
}
