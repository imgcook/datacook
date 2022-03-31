import { Tensor, Tensor1D } from '@tensorflow/tfjs-core';
import { zeros, getDataByType } from '../utils';

/**
 * Encodes an array, Tensor or Danfo Series using unique labels
 * @param {data} data [Array|Series]
 * @returns Array
 */
class LabelEncoder {
  public labels: Array<any>
  private data: Array<any>

  /**
   * Maps data to unique integer labels
   * @param {data} data [Array|Series|Tensor1D]
   * @returns Array
   */
  public async fit(data: Tensor1D[] | Array<any> | any): Promise<any> {
    this.data = await getDataByType(data);
    const dataSet = new Set(this.data);
    this.labels = Array.from(dataSet);
    const label = this.labels;
    const encodedData = this.data.map((x: any) => {
      return label.indexOf(x);
    });

    return encodedData;
  }

  /**
   * Transform data using the label generated from fitting
   * @param {data} data [Array|Series]
   * @returns Array
   */
  public async transform(data: Tensor | Array<any>): Promise<any> {
    const label = this.labels;
    const dataToEncode = await getDataByType(data) || this.data;
    const encodedData = dataToEncode.map((x : any) => {
      return label.indexOf(x);
    });
    return encodedData;
  }
}


/**
 * Encodes an array, Tensor or Danfo Series uaing one-hot encoding format
 * @param {data} data [Array|Series]
 * @returns Array
 */
class OneHotEncoder {
  public labels: Array<any>;
  private data: Tensor1D[] | Array<any>;

  /**
   * Maps data to unique integer labels
   * @param {data} data [Array|Series|Tensor1D]
   * @returns Array
   */
  public async fit(data: Tensor1D[] | Array<any> | any): Promise<any> {
    this.data = await getDataByType(data);
    const data_set = new Set(this.data);
    this.labels = Array.from(data_set);

    const dataLen = this.data.length;

    const encodedData = zeros(dataLen, this.labels.length);

    for (let i = 0; i < dataLen; i++) {
      const elem = this.data[i];
      const elemIndex = this.labels.indexOf(elem);
      encodedData[i][elemIndex] = 1;
    }
    return encodedData;
  }

  /**
   * Transform data using the label generated from fitting
   * @param {data} data [Array|Series]
   * @returns Array
   */
  public async transform(data: Tensor | Array<any> ): Promise<any> {
    const dataToEncode = await getDataByType(data) || this.data;
    const tempLabels = this.labels;
    const encodedData = zeros(dataToEncode.length, tempLabels.length);

    for (let i = 0; i < dataToEncode.length; i++) {
      const elem = dataToEncode[i];
      const elemIndex = tempLabels.indexOf(elem);
      encodedData[i][elemIndex] = 1;
    }

    return encodedData;
  }
}

export {
  LabelEncoder, OneHotEncoder
};

