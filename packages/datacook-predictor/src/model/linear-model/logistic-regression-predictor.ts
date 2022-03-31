import { BaseClassifier } from "../base";
import { Matrix } from 'ml-matrix';
/**
 * Logistic regression predictor
 * */
export class LogisticRegressionPredictor<T extends number | string> extends BaseClassifier<T> {

  private featureSize: number;
  private modelWeights: [ number[][], number[] ];

  /**
   * Make prediction using logistic regression model.
   * @param xData Input feaures
   * @returns predicted classes
   */
  public async predict(xData: number[][]) : Promise<T[]> {
    const x = new Matrix(xData);
    const coefMatrix = new Matrix(this.modelWeights[0]);
    //const biasMatrix = new Matrix(this.modelWeights[1]);
    if (x.columns != this.featureSize) {
      throw TypeError('Feature size does not match');
    }
    const scores = x.mmul(coefMatrix).addRowVector(this.modelWeights[1]).to2DArray();
    return this.getPredClass(scores);
  }
  /**
   * Predict probabilities using logistic regression model.
   * @param xData Input features
   * @returns Predicted probabilities
   */
  public async predictProba(xData: number[][]) : Promise<number[][]> {
    const x = new Matrix(xData);
    const coefMatrix = new Matrix(this.modelWeights[0]);
    if (x.columns != this.featureSize) {
      throw TypeError('Feature size does not match');
    }
    const scores = x.mmul(coefMatrix).addRowVector(this.modelWeights[1]);
    if (this.isBinaryClassification()) {
      const expBase = scores.neg().exp().to2DArray();
      return expBase.map((d: number[]): number[] => [1.0 / (1.0 + d[0])]);
    } else {
      return scores.divColumnVector(scores.exp().sum('row')).to2DArray();
    }
  }

  /**
   * Load model from dumped model json string.
   * @param modelJson Dumped model json string
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<LogisticRegressionPredictor<T>> {
    const modelParams = JSON.parse(modelJson);
    if (modelParams.name !== 'LogisticRegression') {
      throw new RangeError(`${modelParams.name} is not Logistic Regression`);
    }
    const { classes, modelWeights, featureSize } = modelParams;
    this.featureSize = featureSize;
    if (classes) {
      this.initClasses(classes, 'binary-only');
    }
    if (modelWeights?.length) {
      this.modelWeights = modelWeights;
    }
    return this;
  }
}
