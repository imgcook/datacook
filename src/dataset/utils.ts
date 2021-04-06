import { DataAccessor, Dataset, DatasetMeta, Sample } from "./types";
import { range, shuffle } from "../generic";

interface DatasetData<T extends Sample> {
  trainData: Array<T>,
  testData: Array<T>,
  validData?: Array<T>
}

export class DataAccessorImpl<T extends Sample> implements DataAccessor<T> {
  private data: Array<T>;
  private dataIndexes: Array<number>;
  private cursor: number;

  constructor(data: Array<T>) {
    this.data = data;
    this.cursor = 0;
    this.dataIndexes = range(0, data.length);
  }

  shuffle(): void {
    if (this.data.length === 0) this.dataIndexes = [];
    else shuffle(this.dataIndexes);
  }

  async next(): Promise<T | null> {
    return this.data[this.dataIndexes[this.cursor++]];
  }
  async nextBatch(batchSize: number): Promise<Array<T>> {
    const ret: Array<T> = [];

    // return zero-length array if 0 present
    if (batchSize === 0) {
      return ret;
    }

    // return the rest of dataset if -1 present
    if (batchSize === -1) {
      let value = await this.next();
      while (value) {
        ret.push(value);
        value = await this.next();
      }
      return ret;
    }

    if (batchSize < -1) {
      throw new RangeError(`Batch size should be larger than -1 but ${batchSize} is present`);
    }

    // default behaviour
    while (batchSize--) {
      const value = await this.next();
      if (!value) break;
      ret.push(value);
    }
    return ret;
  }
  async seek(offset: number): Promise<void> {
    this.cursor = offset;
  }

}

class DatasetImpl<T extends Sample, D extends DatasetMeta> implements Dataset<T, D> {
  private meta: D;

  public train: DataAccessor<T>;
  public test: DataAccessor<T>;
  public valid?: DataAccessor<T>;

  constructor(datasetData: DatasetData<T>, datasetMeta: D) {
    this.meta = datasetMeta;
    this.train = new DataAccessorImpl(datasetData.trainData);
    this.test = new DataAccessorImpl(datasetData.testData);
    this.valid = datasetData.validData ? new DataAccessorImpl(datasetData.validData) : null;
  }

  async getDatasetMeta() {
    return this.meta;
  }

  shuffle(): void {
    this.train.shuffle();
    this.test.shuffle();
    this.valid?.shuffle();
  }

}

export function makeDataset<T extends Sample, D extends DatasetMeta> (datasetData: DatasetData<T>, datasetMeta: D): Dataset<T, D> {
  return new DatasetImpl(datasetData, datasetMeta);
}

export interface TransformOption<IN_META extends DatasetMeta, IN_SAMPLE extends Sample, OUT_SAMPLE extends Sample = IN_SAMPLE, OUT_META extends DatasetMeta = IN_META> {
  next: (sample: IN_SAMPLE) => Promise<OUT_SAMPLE>,
  metadata: (meta: IN_META) => Promise<OUT_META>
}

function makeTransformDataAccessor<IN extends Sample, OUT extends Sample>(accessor: DataAccessor<IN>, next: (sample: IN) => Promise<OUT>): DataAccessor<OUT> {
  const transformedData: DataAccessor<OUT> = {
    seek: (pos: number) => accessor.seek(pos),
    shuffle: (seed?: string) => accessor.shuffle(seed),
    next: async () => next(await accessor.next()),
    nextBatch: async (batchSize: number) => {
      const samples: Array<OUT> = [];
      while (batchSize--) {
        const s = await this.next();
        if (s) {
          samples.push(s);
        } else {
          break;
        }
      }
      return samples;
    }
  };
  return transformedData;
}

export function transformDataset<IN_META extends DatasetMeta, IN_SAMPLE extends Sample, OUT_SAMPLE extends Sample = IN_SAMPLE, OUT_META extends DatasetMeta = IN_META>
(transformOption: TransformOption<IN_META, IN_SAMPLE, OUT_SAMPLE, OUT_META>, dataset: Dataset<IN_SAMPLE, IN_META>): Dataset<OUT_SAMPLE, OUT_META> {
  const { metadata, next } = transformOption;

  const internalDataset: Dataset<OUT_SAMPLE, OUT_META> = {
    shuffle: (seed?: string) => dataset.shuffle(seed),
    getDatasetMeta: async () => metadata(await dataset.getDatasetMeta()),
    train: makeTransformDataAccessor<IN_SAMPLE, OUT_SAMPLE>(dataset.train, next),
    test: makeTransformDataAccessor<IN_SAMPLE, OUT_SAMPLE>(dataset.test, next)
  };

  return internalDataset;
}

export function transformNextInDataset<IN_META extends DatasetMeta, IN_SAMPLE extends Sample, OUT_SAMPLE extends Sample = IN_SAMPLE>
(next: (sample: IN_SAMPLE) => Promise<OUT_SAMPLE>, dataset: Dataset<IN_SAMPLE, IN_META>) {
  const internalDataset: Dataset<OUT_SAMPLE, IN_META> = {
    shuffle: (seed?: string) => dataset.shuffle(seed),
    getDatasetMeta: () => dataset.getDatasetMeta(),
    train: makeTransformDataAccessor<IN_SAMPLE, OUT_SAMPLE>(dataset.train, next),
    test: makeTransformDataAccessor<IN_SAMPLE, OUT_SAMPLE>(dataset.test, next)
  };

  return internalDataset;
}
