import { BaseEstimator } from '../model/base';

export abstract class TransformerMixin extends BaseEstimator {
  constructor(){
    super();
  }
  abstract transform(X: number[][]): Promise<number[][]>;
  abstract inverseTransform(X: number[][]): Promise<number[][]>;
}
