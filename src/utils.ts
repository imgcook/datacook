import { Tensor, Tensor1D } from '@tensorflow/tfjs-core';

function _zeros(row: number, column: number): Array<any> {
  let zero_array = [];
  for (let i = 0; i < row; i++) {
    let col_data = Array.from(new Uint8Array(column));
    zero_array.push(col_data);
  }
  return zero_array;
}


async function getDataByType(data: Tensor1D[] | Array<any> | any) : Promise<any>{
  if (Array.isArray(data)) {
    return data;
  } else if (data instanceof Tensor) {
    return await data.array();
  } else {
    return undefined;
  }
}


export {
  _zeros, getDataByType
};
