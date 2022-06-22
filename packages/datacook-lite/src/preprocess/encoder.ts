import { unique, oneHot } from "../basic";

export type CateMap = {
  [ key: string ]: number
}

export type OneHotDropTypes = 'first' | 'binary-only' | 'none';
export type OneHotEncoderParams = {
  drop: OneHotDropTypes
}

export abstract class EncoderBase<T extends number | string> {
  public categories: T[];
  public cateMap: CateMap;
  public a: number;
  /**
   * Init encoder
   * @param x data input used to init encoder
   * @param categories user input categories
   */
  public async init(x: T[]): Promise<void> {
    this.categories = unique(x);
    const cateData = this.categories;
    const cateMap: CateMap = {};
    for (let i = 0; i < cateData.length; i++) {
      const key = cateData[i];
      cateMap[`${key}`] = i;
    }
    this.cateMap = cateMap;
  }
  abstract encode(x: T[]): Promise<number[][] | number[]>;
  abstract decode(x: number[][] | number[]): Promise<T[]>;
}

/**
 * Encode categorical features as a one-hot numeric array.
 *
 */
export class OneHotEncoder<T extends number | string> extends EncoderBase<T> {
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
  public async encode(x: T[]): Promise<number[][] | number[]> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    const xInd = x.map((d: T) => this.cateMap[d]);
    const nCate = this.categories.length;
    if (this.drop === 'binary-only' && nCate === 2) {
      return xInd;
    } else if (this.drop === 'first') {
      return oneHot(xInd.map((d: number) => d - 1), nCate - 1);
    } else {
      return oneHot(xInd, nCate);
    }
  }
  /**
   * Decode one-hot array to original category array
   * @param x one-hot format data need to transform
   * @returns transformed category data
   */
  public async decode(x: number[] | number[][]): Promise<T[]> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    const nCate = this.categories.length;
    const codeSize = this.drop === 'first' ? nCate - 1 : this.drop === 'binary-only' && nCate === 2 ? 1 : nCate;
    const shapeCorrect = codeSize > 1
      ? x[0] instanceof Array && x[0]?.length == codeSize
      : x[0] instanceof Array && x[0]?.length == 1 || typeof(x[0]) === 'number';
    if (!shapeCorrect) {
      throw new TypeError('Input shape does not match');
    }
    const cateInd = (this.drop === 'binary-only' && nCate === 2)
      ? x.map((d: number | number[]): number => Number(d instanceof Array ? d[0] > 0.5 : d > 0.5))
      : x.map((d: number[] | number): number => d instanceof Array ? d.indexOf(Math.max(...d)) : 0);
    const cateArrays: T[] = [];
    if (this.drop === 'binary-only' && nCate === 2) {
      cateInd.forEach((ind: number) => {
        if (ind != 0 && ind != 1) {
          throw RangeError('Index out of range');
        }
        cateArrays.push(this.categories[ind]);
      });
    } else if (this.drop === 'first') {
      cateInd.forEach((ind: number, i: number) => {
        const xi = x[i];
        if (xi instanceof Array && xi[ind] === 0) {
          cateArrays.push(this.categories[0]);
        } else {
          cateArrays.push(this.categories[ind + 1]);
        }
      });
    } else {
      cateInd.forEach((ind: number) => {
        cateArrays.push(this.categories[ind]);
      });
    }
    return cateArrays;
  }
}

export class LabelEncoder<T extends number | string> extends EncoderBase<T> {
  /**
   * Encode a given feature into one-hot format
   * @param x feature array need to encode
   * @returns transformed one-hot feature
   */
  public async encode(x: T[]): Promise<number[]> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    return x.map((d: number|string) => this.cateMap[d]);
  }
  /**
   * Decode a label one-hot array to original category array
   * @param x encoded data need to transform
   * @returns transformed category data
   */
  public async decode(x: number[]): Promise<T[]> {
    if (!this.categories) {
      throw TypeError('Please init encoder using init()');
    }
    const cateArrays: T[] = [];
    x.forEach((ind: number) => cateArrays.push(this.categories[ind]));
    return cateArrays;
  }
}
