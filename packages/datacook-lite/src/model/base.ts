import { OneHotEncoder, OneHotDropTypes } from "../preprocess/encoder";
import { checkJsArray2D } from "../utils";

class BaseEstimator {
  public nFeature: number;
  public estimatorType: string;
  /**
   * Check if input feature match the required feature size. If reset is true,
   * reset nFeature = x.shape[1]
   *
   * @param x Input feature of shape (nSample, nFeatures).
   * @param reset if true, the `nFeatures` attribute is set to `x.shape[1]`.
   */
  public checkAndSetNFeatures(x: number[][], reset: boolean): void {
    checkJsArray2D(x);
    const featureCount = x[0].length;
    if (!reset && featureCount !== this.nFeature) {
      throw new TypeError(`X has ${featureCount} features, but is expected to have ${this.nFeature} features as input.`);
    }
    if (reset || !this.nFeature) {
      this.nFeature = x[0].length;
    }
  }
  public isClassifier(): boolean {
    return this.estimatorType === 'classifier';
  }
}

class BaseClustring extends BaseEstimator {
  public estimatorType = 'clustering';
  public validateData(x: number[][], reset = false): void {
    checkJsArray2D(x);
    this.checkAndSetNFeatures(x, reset);
  }
}

class BaseClassifier<T extends number | string> extends BaseEstimator{
    public estimatorType = 'classifier';
    public classOneHotEncoder: OneHotEncoder<T>;
    // get label one-hot expression
    public async getLabelOneHot(y: T[]): Promise<number[][] | number[]> {
      return await this.classOneHotEncoder.encode(y);
    }

    public async initClasses(y: T[], drop: OneHotDropTypes = 'none'): Promise<void> {
      this.classOneHotEncoder = new OneHotEncoder({ drop });
      await this.classOneHotEncoder.init(y);
    }

    public classes(): T[] {
      return this.classOneHotEncoder?.categories;
    }

    public isBinaryClassification(): boolean {
      return this.classes().length === 2;
    }

    public async getPredClass(score: number[][]): Promise<T[]> {
      return await this.classOneHotEncoder.decode(score);
    }

    public validateData(x: number[][], y: T[]): { x: number[][], y: T[] } {
      const xCount = x.length;
      const yCount = y.length;
      if (xCount != yCount) {
        throw new RangeError(
          'the size of training set and training labels must be the same.'
        );
      }
      return { x, y };
    }
}

export { BaseClassifier, BaseEstimator, BaseClustring };
