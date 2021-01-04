import { Tensor, Tensor1D } from "@tensorflow/tfjs-core";
import { Series } from "danfojs-node";

function _zeros(row: number, column: number): Array<any> {
  let zero_array = [];

  for (let i = 0; i < row; i++) {
    let col_data = Array(column);
    for (let j = 0; j < column; j++) {
      col_data[j] = 0;
    }
    zero_array.push(col_data);
  }
  return zero_array;
}


function getDataByType(data: Tensor1D[] | Array<any> | Series | any) : any{
  if (Array.isArray(data)) {
    return data;
  } else if (data instanceof Series) {
    return data.values;
  } else if (data instanceof Tensor) {
    return data.arraySync();
  } else {
    return undefined;
  }
}


export {
  _zeros, getDataByType
};
