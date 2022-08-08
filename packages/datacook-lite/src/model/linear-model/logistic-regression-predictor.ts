import { BaseClassifier } from "../base";
import { matrix, vector, Vector, Matrix } from "../../core/classes";
import { add2d, matMul2d, softmax2d, sigmoid2d } from "../../core/op";

/**
 * Logistic regression predictor
 */
export class LogisticRegressionPredictor extends BaseClassifier<number | string> {

  private featureSize: number;
  private modelWeights: [ Matrix, Vector ];

  /**
   * Make prediction using logistic regression model.
   * @param xData Input feaures
   * @returns predicted classes
   */
  public async predict(xData: number[][]) : Promise<(string | number | boolean)[]> {
    const x = matrix(xData);
    const scores = add2d(matMul2d(x, this.modelWeights[0]), this.modelWeights[1], 0);
    return this.getPredClass(scores.data);
  }
  /**
   * Predict probabilities using logistic regression model.
   * @param xData Input features
   * @returns Predicted probabilities
   */
  public async predictProba(xData: number[][]) : Promise<number[][]> {
    const x = matrix(xData);
    if (x.shape[1] !== this.featureSize) {
      throw TypeError('Feature size does not match');
    }
    const scores = add2d(matMul2d(x, this.modelWeights[0]), this.modelWeights[1], 0);
    return this.isBinaryClassification() ? sigmoid2d(scores).data : softmax2d(scores).data;
  }

  /**
   * Load model from dumped model json string.
   * @param modelJson Dumped model json string
   * @returns model itself
   */
  public async fromJson(modelJson: string): Promise<LogisticRegressionPredictor> {
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
      this.modelWeights = [ matrix(modelWeights[0]), vector(modelWeights[1]) ];
    }
    return this;
  }
}
