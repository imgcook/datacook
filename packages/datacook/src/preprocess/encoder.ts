import { Tensor, unique, oneHot, cast, tensor, argMax, reshape, slice, stack, sub, squeeze, greaterEqual, topk, Tensor1D, tidy } from '@tensorflow/tfjs-core';
import { checkArray } from '../utils/validation';
import { checkShape } from '../linalg/utils';

export type CateMap = {
  [ key: string ]: number
}

export type OneHotDropTypes = 'first' | 'binary-only' | 'none';
export type OneHotEncoderParams = {
  drop: OneHotDropTypes
}

export abstract class EncoderBase {
  public categories: Tensor;
  public cateMap: CateMap;
  /**
   * Init encoder
   * @param x data input used to init encoder
   * @param categories user input categories
   */
  public async init(x: Tensor | number[] | string[] | boolean[]): Promise<void> {
    const { values } = unique(x);
    if (values.dtype === 'int32' || values.dtype === 'float32') {
      this.categories = topk(values, values.shape[0], false).values;
    } else if (values.dtype === 'bool') {
      this.categories = tensor([ false, true ]);
    } else {
      this.categories = values;
    }
    const cateData = await this.categories.data();
    const cateMap: CateMap = {};
    for (let i = 0; i < cateData.length; i++) {
      const key = cateData[i];
      cateMap[key] = i;
    }
    this.cateMap = cateMap;
  }
  abstract encode(x: Tensor | number[] | string[]): Promise<Tensor>;
  abstract decode(x: Tensor): Promise<Tensor>;
}

/**
 * Encode categorical features as a one-hot numeric array.
 *
 */
export class OneHotEncoder extends EncoderBase {
  public drop: OneHotDropTypes;

  /**
   * @param params one-hot encoder parameters
   *
   * Options in params
   * ---------
   * drop: { 'none', 'binary-only', 'first' }
   *    Specifies a method to drop one of the categories per feature,
   *    which is useful to avoid collinear problem.
   *
   *    However, dropping one category may introduce a bias term in
   *    downstream models. For instance in linear regression model,
   *    the effect of the dropped feature might appear in intercept term.
   *
   *    - 'none': default, return all features
   *    - 'first': drop the first categories in each feature
   *    - 'binary-only': drop the first category in each feature with two
   *       categories.
   */
  public constructor (params: OneHotEncoderParams = { drop: 'none' }) {
    super();
    const { drop } = params;
    this.drop = drop;
  }

  /**
   * Encode a given feature into one-hot format
   * @param x feature array need to encode
   * @returns transformed one-hot feature
   */
  public async encode(x: Tensor | number[] | string[] | boolean[]): Promise<Tensor> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    return tidy(() => {
      const xTensor = checkArray(x, 'any', 1);
      const xData = xTensor.dataSync();
      const nCate = this.categories.shape[0];
      const xInd = xData.map((d: number|string) => this.cateMap[d]);
      if (this.drop === 'binary-only' && nCate === 2) {
        return tensor(xInd);
      } else if (this.drop === 'first') {
        return oneHot(cast(sub(tensor(xInd), 1), 'int32'), nCate - 1);
      } else {
        return oneHot(xInd, nCate);
      }
    });
  }
  /**
   * Decode one-hot array to original category array
   * @param x one-hot format data need to transform
   * @returns transformed category data
   */
  public async decode(x: Tensor): Promise<Tensor> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    return tidy(() => {
      const nCate = this.categories.shape[0];
      const codeSize = this.drop === 'first' ? nCate - 1 : this.drop === 'binary-only' && nCate === 2 ? 1 : nCate;
      const shapeCorrect = codeSize > 1 ? checkShape(x, [ -1, codeSize ]) : (checkShape(x, [ -1 ]) || checkShape(x, [ -1, 1 ]));
      if (!shapeCorrect) {
        throw new TypeError('Input shape does not match');
      }
      const cateInd = (this.drop === 'binary-only' && nCate === 2) ? greaterEqual(squeeze(x), 0.5).dataSync() : argMax(x, 1).dataSync();
      const cateTensors: Tensor[] = [];
      if (this.drop === 'binary-only' && nCate === 2) {
        cateInd.forEach((ind: number) => {
          if (ind != 0 && ind != 1) {
            throw RangeError('Index out of range');
          }
          cateTensors.push(slice(this.categories, ind, 1));
        });
      } else if (this.drop === 'first') {
        cateInd.forEach((ind: number, i: number) => {
          if (Number(slice(x, [ i, ind ], [ 1, 1 ]).dataSync()) === 0) {
            cateTensors.push(slice(this.categories, 0, 1));
          } else {
            cateTensors.push(slice(this.categories, ind + 1, 1));
          }
        });
      } else {
        cateInd.forEach((ind: number) => {
          cateTensors.push(slice(this.categories, ind, 1));
        });
      }
      return reshape(stack(cateTensors), [ -1 ]);
    });
  }
}

export class LabelEncoder extends EncoderBase {
  /**
   * Encode a given feature into one-hot format
   * @param x feature array need to encode
   * @returns transformed one-hot feature
   */
  public async encode(x: Tensor | number[] | string[] | boolean[]): Promise<Tensor> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    const xTensor = checkArray(x, 'any', 1);
    const xData = await xTensor.data();
    xTensor.dispose();
    return tensor(xData.map((d: number|string) => this.cateMap[d]));
  }
  /**
   * Decode a label one-hot array to original category array
   * @param x encoded data need to transform
   * @returns transformed category data
   */
  public async decode(x: Tensor | number[]): Promise<Tensor> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    const xData: number[] = x instanceof Tensor ? await (x as Tensor1D).array() : x;
    const cateTensors: Tensor[] = [];
    xData.forEach((ind: number) => {
      cateTensors.push(slice(this.categories, ind, 1));
    });
    return tidy(() => {
      return reshape(stack(cateTensors), [ -1 ]);
    });
  }
}
