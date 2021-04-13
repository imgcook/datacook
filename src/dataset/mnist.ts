
import { DataAccessor, Dataset, DatasetMeta, DatasetSize, DatasetType, ImageDatasetMeta, ImageDimension, Sample } from './types';
import fetch from 'cross-fetch';
import { Buffer } from 'buffer/';
import { DataAccessorImpl } from './utils';
import { range } from '../generic';

const URLs = {
  trainLabel: 'http://pipcook.oss-cn-hangzhou.aliyuncs.com/dataset/mnist/train-labels-idx1-ubyte',
  trainData: 'http://pipcook.oss-cn-hangzhou.aliyuncs.com/dataset/mnist/train-images-idx3-ubyte',
  testData: 'http://pipcook.oss-cn-hangzhou.aliyuncs.com/dataset/mnist/t10k-images-idx3-ubyte',
  testLabel: 'http://pipcook.oss-cn-hangzhou.aliyuncs.com/dataset/mnist/t10k-labels-idx1-ubyte'
};

const DATA_MAGIC = 2051;
const LABEL_MAGIC = 2049;

type mnistSample = Sample<Uint8ClampedArray, number>;
class MNIST implements Dataset<mnistSample, ImageDatasetMeta> {
  private dataMap: Record<string, ArrayBuffer>;
  private trainSamples: Array<mnistSample>;
  private testSamples: Array<mnistSample>;
  private width: number;
  private height: number;

  public test: DataAccessor<mnistSample>;
  public train: DataAccessor<mnistSample>;

  constructor(dataMap: Record<string, ArrayBuffer>) {
    this.dataMap = dataMap;
    this.process();
    this.test = new DataAccessorImpl(this.testSamples);
    this.train = new DataAccessorImpl(this.trainSamples);
  }

  shuffle(seed?: string): void {
    this.test.shuffle(seed);
    this.train.shuffle(seed);
  }

  async getDatasetMeta() {
    const datasetSize: DatasetSize = {
      test: this.trainSamples.length,
      train: this.testSamples.length
    };

    const imageDims: ImageDimension = {
      x: this.width,
      y: this.height,
      z: 1
    };

    const labelMap = range(0, 10).reduce((prev, curr) => {
      prev[curr] = curr.toString();
      return prev;
    }, {} as Record<number, string>);

    const meta: ImageDatasetMeta = {
      type: DatasetType.Image,
      size: datasetSize,
      dimension: imageDims,
      labelMap
    };

    return meta;
  }

  private process() {
    if (! this.dataMap) throw new Error('MNIST should contain data; Consider init with getMNIST');
    this.trainSamples = this.processInternal(this.dataMap['trainData'], this.dataMap['trainLabel']);
    this.testSamples = this.processInternal(this.dataMap['testData'], this.dataMap['testLabel']);
  }

  private processInternal(rawData: ArrayBuffer, rawLabel: ArrayBuffer) {
    const rawDataBuffer = Buffer.from(rawData);
    const rawLabelBuffer = Buffer.from(rawLabel);
    const dataMagic = rawDataBuffer.readInt32BE(0);
    if (dataMagic !== DATA_MAGIC) {
      throw new TypeError('Data header mismatch; aborting');
    }
    const labelMagic = rawLabelBuffer.readInt32BE(0);
    if (labelMagic !== LABEL_MAGIC) {
      throw new TypeError('Label header mismatch; aborting');
    }

    const samples = rawDataBuffer.readInt32BE(4);
    if (samples !== rawLabelBuffer.readInt32BE(4)) {
      throw new TypeError('sample size mismatch; aborting');
    }

    const rowNum = rawDataBuffer.readInt32BE(8);
    const columnNum = rawDataBuffer.readInt32BE(12);
    this.height = rowNum;
    this.width = columnNum;
    const totalPixels = rowNum * columnNum;

    const ret: Array<mnistSample> = new Array(samples);

    for (let i = 0; i < samples; i++) {
      const label = rawLabelBuffer[i + 8];
      const pixels: Uint8ClampedArray = new Uint8ClampedArray(totalPixels);
      for (let y = 0; y < columnNum; y++) {
        for (let x = 0; x < rowNum; x++) {
          pixels[x + y * columnNum] = rawDataBuffer[16 + i * totalPixels + (x + y * columnNum)];
        }
      }
      ret[i] = {
        label,
        data: pixels
      };
    }

    return ret;
  }

  static async getMNIST() {
    console.log('start downloading dataset');
    const fetchData = await Promise.all(Object.values(URLs).map((url) => fetch(url)));
    const abData = await Promise.all(fetchData.map((it) => it.arrayBuffer()));
    const abDataMap = Object.keys(URLs).reduce((prev, curr, idx) => {
      prev[curr] = abData[idx];
      return prev;
    }, {} as Record<string, ArrayBuffer>);

    return new MNIST(abDataMap);
  }
}

export default MNIST;
