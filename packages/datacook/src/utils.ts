import { Tensor, Tensor1D } from '@tensorflow/tfjs-core';

function zeros(row: number, column: number): Array<number[]> {
  const zeroArray = [];
  for (let i = 0; i < row; i++) {
    const colData = Array.from(new Uint8Array(column));
    zeroArray.push(colData);
  }
  return zeroArray;
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
  zeros, getDataByType, sizeFromShape
};
