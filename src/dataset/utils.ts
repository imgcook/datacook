import { DataAccessor, Dataset, DatasetMeta, Sample } from "./types";

interface DatasetData<T extends Sample> {
  trainData: Array<T>,
  testData: Array<T>,
  validData?: Array<T>
}

export class DataAccessorImpl<T extends Sample> implements DataAccessor<T> {
  private data: Array<T>;
  private cursor: number;

  constructor(data: Array<T>) {
    this.data = data;
    this.cursor = 0;
  }

  async next(): Promise<T | null> {
    return this.data[this.cursor++];
  }
  async nextBatch(batchSize: number): Promise<Array<T> | null> {
    const ret = [];
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
  public valid: DataAccessor<T>;

  constructor(datasetData: DatasetData<T>, datasetMeta: D) {
    this.meta = datasetMeta;
    this.train = new DataAccessorImpl(datasetData.trainData);
    this.test = new DataAccessorImpl(datasetData.testData);
    this.valid = new DataAccessorImpl(datasetData.validData);
  }

  async getDatasetMeta() {
    return this.meta;
  }

}

export function makeDataset<T extends Sample, D extends DatasetMeta> (datasetData: DatasetData<T>, datasetMeta: D): Dataset<T, D> {
  return new DatasetImpl(datasetData, datasetMeta);
}
