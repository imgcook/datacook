import { TextDecoder } from 'util';
import { Tensor, tensor } from '@tensorflow/tfjs-core';

type DataType = 'uint8' | 'uint16' | 'int8' | 'int16' | 'uint32' | 'int32' | 'int64' | 'uint64' | 'float32' | 'float64';
type RawDataType = '<u1' | '|u1' | '<u2' | '|i1' | '<i2' | '<u4' | '<i4' | '<u8' | '<i8' | '<f4' | '<f8';
type ArrayTypeConstructor = Float32ArrayConstructor | Float64ArrayConstructor | Int16ArrayConstructor |
                            Uint8ArrayConstructor | Uint16ArrayConstructor | Int8ArrayConstructor |
                            Int32ArrayConstructor | BigUint64ArrayConstructor | BigInt64ArrayConstructor;
type ArrayType = Float32Array | Float64Array | Int16Array | Uint8Array | Uint16Array | Int8Array | Int32Array | BigUint64Array | BigInt64Array;

interface numpy {
  dtype: DataType,
  shape: Array<number>,
  data: ArrayType
}

interface DTypes {
  name: DataType,
  size: number,
  arrayConstructor: ArrayTypeConstructor
}

const dtypes: Record<RawDataType, DTypes> = {
  '<u1': {
    name: 'uint8',
    size: 8,
    arrayConstructor: Uint8Array
  },
  '|u1': {
    name: 'uint8',
    size: 8,
    arrayConstructor: Uint8Array
  },
  '<u2': {
    name: 'uint16',
    size: 16,
    arrayConstructor: Uint16Array
  },
  '|i1': {
    name: 'int8',
    size: 8,
    arrayConstructor: Int8Array
  },
  '<i2': {
    name: 'int16',
    size: 16,
    arrayConstructor: Int16Array
  },
  '<u4': {
    name: 'uint32',
    size: 32,
    arrayConstructor: Int32Array
  },
  '<i4': {
    name: 'int32',
    size: 32,
    arrayConstructor: Int32Array
  },
  '<u8': {
    name: 'uint64',
    size: 64,
    arrayConstructor: BigUint64Array
  },
  '<i8': {
    name: 'int64',
    size: 64,
    arrayConstructor: BigInt64Array
  },
  '<f4': {
    name: 'float32',
    size: 32,
    arrayConstructor: Float32Array
  },
  '<f8': {
    name: 'float64',
    size: 64,
    arrayConstructor: Float64Array
  }
};

const MAGIC = 'NUMPY';

/**
 * Modified from https://github.com/aplbrain/npyjs/blob/master/index.js
 * @param arrayBuffer npy arrayBuffer
 */
export function parse(arrayBuffer: ArrayBuffer): numpy {
  const magic = String.fromCharCode.apply(null, new Uint8Array(arrayBuffer.slice(1, 6)));

  if (magic != MAGIC) {
    throw new Error('input is not a npy file');
  }

  const headerLength = new DataView(arrayBuffer.slice(8, 10)).getUint8(0);
  const dataOffset = 10 + headerLength;

  let headerContent: string = new TextDecoder('utf-8').decode(
    new Uint8Array(arrayBuffer.slice(10, headerLength + 10))
  );
  const header = JSON.parse(
    headerContent
      .replace(/'/g, '"')
      .replace('False', 'false')
      .replace('(', '[')
      .replace(/,*\),*/g, ']')
  );

  const shape = header.shape;
  const descr: RawDataType = header.descr;
  const dtype = dtypes[descr];
  const data = new dtype['arrayConstructor'](
    arrayBuffer,
    dataOffset
  );

  return {
    dtype: dtype.name,
    data,
    shape
  };
}

export function parse2Tensor(arrayBuffer: ArrayBuffer): Tensor {
  const numpy = parse(arrayBuffer);
  if (numpy.data instanceof Float64Array || numpy.data instanceof BigInt64Array || numpy.data instanceof BigUint64Array) {
    throw new Error(`tfjs does not support ${numpy.dtype} at this time`);
  }
  const retTensor = tensor(numpy.data, numpy.shape);
  return retTensor;
}
