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
