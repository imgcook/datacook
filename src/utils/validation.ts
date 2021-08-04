import { Tensor, cast, tensor, DataType } from '@tensorflow/tfjs-core';

export function checkArray(array: Tensor | Array<any>, dtype: DataType = 'float32', ensureDimension: number = -1, allowNaN: boolean = true): Tensor{
    if (array instanceof Array){
        array  = tensor(array);
    }
    const dim = array.rank;
    const arr_dtype = array.dtype;
    if (ensureDimension != -1 && dim != ensureDimension){
        throw new TypeError(`Dimension of input require to be ${ensureDimension} but receive ${dim}`);
    }
    if (dtype != arr_dtype){
        array = cast(array, dtype);
    }
    return array
}