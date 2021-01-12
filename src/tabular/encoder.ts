import { Tensor, Tensor1D } from "@tensorflow/tfjs-core";
import { _zeros, getDataByType } from "../utils";

/**
 * Encodes an array, Tensor or Danfo Series using unique labels
 * @param {data} data [Array|Series]
 * @returns Array
 */
class LabelEncoder {
  public labels: Array<any>
  private _data: Array<any>

  constructor(data: Tensor1D[] | Array<any> | any){
    if (Array.isArray(data)) {
      this._data = data;
    } else if (data instanceof Tensor) {
      this._data = data.arraySync();
    } else {
      throw new TypeError("data must be one of Array or Tensor1D");
    }
  }

  /**
   * Maps data to unique integer labels
   * @param {data} data [Array|Series|Tensor1D]
   * @returns Array
   */
  public fit(): Array<any> {
    const data_set = new Set(this._data);
    this.labels = Array.from(data_set);
    const _label = this.labels;
    let encoded_data = this._data.map((x: any) => {
      return _label.indexOf(x);
    });

    return encoded_data;
  }

  /**
   * Transform data using the label generated from fitting
   * @param {data} data [Array|Series]
   * @returns Array
   */
  public transform(data: Tensor | Array<any>): Array<any> {
    const _label = this.labels;
    const data_to_encode = getDataByType(data) || this._data;
    let encoded_data = data_to_encode.map((x : any) => {
      return _label.indexOf(x);
    });
    return encoded_data;
  }
}


/**
 * Encodes an array, Tensor or Danfo Series uaing one-hot encoding format
 * @param {data} data [Array|Series]
 * @returns Array
 */
class OneHotEncoder {
  public labels: Array<any>
  private _data: Tensor1D[] | Array<any>

  constructor(data: Tensor1D[] | Array<any> | any ){
    if (Array.isArray(data)) {
      this._data = data;
    } else if (data instanceof Tensor) {
      this._data = data.arraySync();
    } else {
      throw new TypeError("data must be one of Array or Tensor1D");
    }
  }

  /**
   * Maps data to unique integer labels
   * @param {data} data [Array|Series|Tensor1D]
   * @returns Array
   */
  public fit(): Array<any> {
    const data_len = this._data.length;
    const data_set = new Set(this._data);
    this.labels = Array.from(data_set);

    let encoded_data = _zeros(data_len, this.labels.length);
    for (let i = 0; i < data_len; i++) {
      let elem = this._data[i];
      let elem_index = this.labels.indexOf(elem);
      encoded_data[i][elem_index] = 1;
    }
    return encoded_data;
  }

  /**
   * Transform data using the label generated from fitting
   * @param {data} data [Array|Series]
   * @returns Array
   */
  public transform(data: Tensor | Array<any> ): Array<any> {
    const data_to_encode = getDataByType(data) || this._data;
    let encoded_data = _zeros(data_to_encode.length, this.labels.length);

    for (let i = 0; i < data_to_encode.length; i++) {
      let elem = data_to_encode[i];
      let elem_index = this.labels.indexOf(elem);
      encoded_data[i][elem_index] = 1;
    }

    return encoded_data;
  }
}

export {
  LabelEncoder, OneHotEncoder
};

