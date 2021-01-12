import { Tensor, Tensor1D } from "@tensorflow/tfjs-core";

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


function getDataByType(data: Tensor1D[] | Array<any> | any) : Array<any>{
  if (Array.isArray(data)) {
    return data;
  } else if (data instanceof Tensor) {
    return data.arraySync();
  } else {
    return undefined;
  }
}


export {
  _zeros, getDataByType
};
