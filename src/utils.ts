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

function sizeFromShape(shape: number[]): number {
  if (shape.length === 0) {
    // Scalar.
    return 1;
  }
  let size = shape[0];
  for (let i = 1; i < shape.length; i++) {
    size *= shape[i];
  }
  return size;
}

export {
  _zeros, getDataByType, sizeFromShape
};
