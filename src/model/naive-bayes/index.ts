
import { BaseClassifier } from "../base";
import { Tensor, Tensor1D, zeros, oneHot, unique, tensor1d, add, sub, log, mul, argMax, cast, squeeze, slice, exp, reshape,
  matMul, transpose, sum, div, divNoNan, booleanMaskAsync, gather, mean, stack, Tensor2D, tensor } 
  from '@tensorflow/tfjs-core';
//import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
//import '@tensorflow/tfjs-core/src/public/chained_ops/register_all_chained_ops';


type ClassMap = {
  [key: string]: number
}

/**
 * Multinomial naive bayes classifier
 */
export class MultinomialNB extends BaseClassifier {

  private conditionProb: Tensor;
  private priorProb: Tensor;
  private classes: Tensor;
  public alpha: number;
  public featureCount: Tensor;
  public classCount: Tensor;
  public classMap: ClassMap;

  constructor (alpha: number = 1.0) {
    super();
    this.alpha = alpha;
  }

  private updateClassLogPrior() {
    this.classCount.print();
    if (this.classCount) {
      const nData = cast(sum(this.classCount),'float32');
      this.priorProb = sub(log(this.classCount), log(nData));
      log(this.classCount).print();
      log(nData).print();
      sub(log(this.classCount), log(nData)).print();
      nData.print();
      console.log(nData.dtype);
      this.classCount.print();
      div(this.classCount, nData).print();
      this.priorProb = log(div(this.classCount, nData))
      this.priorProb.print();
    } 

  }

  private updateFeatureLogProb() {
    if (this.featureCount){
      const axis_h = 1;
      const featureCountFixed = add(this.featureCount, this.alpha);
      const classCountFixed = reshape(sum(featureCountFixed, axis_h),[-1, 1]);
      
      this.conditionProb = sub(log(featureCountFixed), log(classCountFixed));
    }
  }

  private updateFeautrCount(featureCount: Tensor) {
    featureCount = cast(featureCount, 'float32');
    if (this.featureCount){
      this.featureCount = add(this.featureCount, featureCount);
    }else{
      this.featureCount = featureCount;
    }
  }

  private updateClassCount(classCount: Tensor) {
    classCount = cast(classCount, 'float32');
    if (this.classCount) {
      this.classCount = add(this.classCount, classCount);
    }else{
      this.classCount = classCount;
    }
  }

  private getLogLikelihood(X: Tensor2D): Tensor {
    const v = matMul(X, transpose(this.conditionProb));
    const predictions = add(v, this.priorProb);
    return predictions;
  }

  // check if first call to train
  private firstCall(): boolean{
    if(this.featureCount && this.priorProb && this.conditionProb && this.classCount){
      return false;
    }
    return true;
  }

  // get class map
  private updateClassMap(){
    if (this.classes){
      const classData = this.classes.dataSync();
      let classMap: ClassMap = {}
      for(let i=0; i < classData.length; i++){
        const key = classData[i];
        classMap[key] = i;
      }
      this.classMap = classMap;
    }
  }

  // get one-hot vector for new input data
  private getNewBatchOneHot(y: Tensor): Tensor {
    const yData = y.dataSync();
    const yInd = yData.map((d) => { return this.classMap[d] });
    return tensor(yInd);
  }

  public async trainBatch(XData: Array<any> | Tensor, yData: Array<any> | Tensor) {
    
    const { X, y } = this.validateData(XData, yData);
    const { values, indices } = unique(y);
    const nClass =  values.shape[0];
    const firstCall = this.firstCall();
    let yOneHot;

    if (firstCall){
      this.classes = values;
      this.updateClassMap();
      yOneHot = oneHot(indices, nClass);
      
    }
    else{
      const yInd = this.getNewBatchOneHot(y);
      const nFeatures = X.shape[1];
      if (nFeatures != this.featureCount.shape[1]){
        throw new Error('feature size does not match to previous training dataset');
      }
      yOneHot = oneHot(yInd, nClass);
    }

    yOneHot.print();
    const axis_h = 0;
    // update class count
    const classCount = sum(yOneHot, axis_h);
    classCount.print();
    this.updateClassCount(classCount);
    this.classCount.print();

    // update feature count
    const featureCounts = []
    for (let i = 0; i < nClass ; i++) {
      const axis_h = 1;
      const axis_v = 0;
      const classMask = squeeze(cast(gather(yOneHot, [i], axis_h), 'bool'));
      X.print();
      classMask.print();
      const data_i = await booleanMaskAsync(X, classMask);
      const featureCount = sum(data_i, axis_v);
      featureCounts.push(featureCount);
    }
    const featureCountTensor = stack(featureCounts);
    this.updateFeautrCount(featureCountTensor);
    this.featureCount.print();
    // update prior prob
    
    this.updateClassLogPrior();
  
    // update feature conditional log prob
    this.updateFeatureLogProb();
    console.log('class prior')
    this.priorProb.print();
    console.log('condition prob')
    this.conditionProb.print();
  }

  /**
   * Training multinomial naive bayes model according to X, y.
   * @param XData feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features)
   * @param yData label array, one dimension numeric array or 1D Tensor of size n_samples, use different integer values to represent different labels
   * @returns classifier itself
   */
  public async train(XData: Array<any> | Tensor, yData: Array<any> | Tensor) {
    
    const { X, y } = this.validateData(XData, yData);

    const { values, indices } = unique(y);
    this.classes = values;

    const nData = y.shape[0];
    const nFeatures = X.shape[1];
    const nClass =  values.shape[0];
    const yInt  = cast(y, 'int32');
    const yOneHot = oneHot(yInt, nClass);
    const axis = 0;
    const classSum = sum(yOneHot, axis);
    const priorProb = div(classSum, nData);
    const priorProbLog = log(priorProb);

    this.priorProb = priorProbLog;
    
    
    // compute log conditional probability
    let conditionProbs = [];

    priorProb.print();
    
    for (let i = 0; i < nClass ; i++) {
      const axis_h = 1;
      const axis_v = 0;
      const classMask = squeeze(cast(gather(yOneHot, [i], axis_h), 'bool'));
      X.print();
      classMask.print();
      const data_i = await booleanMaskAsync(X, classMask);
      const data_i_fix = add(data_i, 1);
      
      const sum_data_i = reshape(sum(data_i_fix, axis_h), [-1,1]);
      //const sum_data_i_fix = add(sum_data_i, nFeatures);

      const probs = divNoNan(data_i_fix, sum_data_i);
      sum_data_i.print();
      probs.print();
      const conditionProb_i = mean(probs, axis_v);
      conditionProb_i.print();
      conditionProbs.push(conditionProb_i);
    }

    const conditionProbsTensor = stack(conditionProbs);
    conditionProbsTensor.print();
    const conditionProbsTensorLog = log(conditionProbsTensor);
    this.conditionProb = conditionProbsTensorLog;
    conditionProbsTensorLog.print();
    return this;
  }

  /**
   * Perform classification on an array of test vector X.
   * @param X input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 1D Tensor of shape n_samples, Predicted target value of X.
   */
  public predict(X: Tensor2D): Tensor {
    const logLikelihood = this.getLogLikelihood(X);
    logLikelihood .print();
    const axis_h = 1;
    const classInd = argMax(logLikelihood, axis_h);
    //const classVal = slice(this.values, classInd, [1]);
    const classVal = gather(this.classes, classInd);
    return classVal;
  }

  /**
   * Return probablity estimates for test vector X.
   * @param X input feature array, two dimension numeric array or 2D Tensor of shape (n_samples, n_features).
   * @returns 2D Tensor of shape (n_samples, n_claases). Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order
   */
  public predictProba(X: Tensor2D): Tensor {
    const axis_h = 1;
    const logLikelihood = this.getLogLikelihood(X);
    const likeliHood = exp(logLikelihood);
    const sumLikelihood = reshape(sum(likeliHood, axis_h), [-1,1]);
    likeliHood.print();
    sumLikelihood.print();
    const proba = div(likeliHood, sumLikelihood);
    return proba;
  }

  public load(model:any) {
    if (model.name !== 'MultinomialNB') {
      throw new RangeError(`${model.name} is not a Multinomial Naive Bayes`);
    }
    return new MultinomialNB(model);
  }

  public toJson() {
    return {
      name: 'MultinomialNB',
      priorProb: this.priorProb.arraySync(),
      conditionProb: this.conditionProb.arraySync()
    }
  }

}

