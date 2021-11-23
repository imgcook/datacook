import { RecursiveArray, Tensor } from "@tensorflow/tfjs-core";
import { BaseEstimator } from "../model/base";

export abstract class TransformerMixin extends BaseEstimator {
  constructor(){
    super();
  }
  public async fitTransform(X: Tensor | RecursiveArray<number>): Promise<Tensor> {
    await this.fit(X);
    return await this.transform(X);
  }
  abstract fit(X: Tensor | RecursiveArray<number>): void;
  abstract transform(X: Tensor | RecursiveArray<number>): Promise<Tensor>;
  abstract inverseTransform(X: Tensor | RecursiveArray<number>): Promise<Tensor>;
}
