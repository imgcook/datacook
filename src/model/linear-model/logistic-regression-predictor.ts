import { RecursiveArray, Tensor, matMul, tensor, add, sigmoid, softmax } from "@tensorflow/tfjs-core";
import { checkShape } from "../../linalg/utils";
import { checkArray } from "../../utils/validation";
import { BaseClassifier } from "../base";
import '@tensorflow/tfjs-backend-cpu';
/**
 * Logistic regression predictor
 * */
export class LogisticRegressionPredictor extends BaseClassifier {

  private featureSize: number;
  private modelWeights: RecursiveArray<number>;

  /**
   * Make prediction using logistic regression model.
   * @param xData Input feaures
   * @returns predicted classes
   */
  public async predict(xData: Tensor | RecursiveArray<number>) : Promise<Tensor> {
    const x = checkArray(xData, 'float32');
    const scores = add(matMul(x, tensor(this.modelWeights[0])), this.modelWeights[1]);
    return await this.getPredClass(scores);
  }
  /**
   * Predict probabilities using logistic regression model.
   * @param xData Input features
   * @returns Predicted probabilities
   */
  public async predictProba(xData: Tensor | RecursiveArray<number>) : Promise<Tensor> {
    const x = checkArray(xData, 'float32');
    if (!checkShape(x, [ -1, this.featureSize ])) {
      throw TypeError('Feature size does not match');
    }
    const scores = add(matMul(x, tensor(this.modelWeights[0])), this.modelWeights[1]);
    return this.isBinaryClassification() ? sigmoid(scores) : softmax(scores);
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
      this.modelWeights = modelWeights;
    }
    return this;
  }
}
