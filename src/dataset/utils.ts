import { Dataset, Sample } from "./types";
import { range, shuffle } from "../generic";

export class ArrayDatasetImpl<T extends Sample> implements Dataset<T> {
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

export function makeTransform<IN extends Sample, OUT extends Sample>(dataset: Dataset<IN>, transform: (sample: IN) => Promise<OUT>): Dataset<OUT> {
  const transformedData: Dataset<OUT> = {
    seek: (pos: number) => dataset.seek(pos),
    shuffle: (seed?: string) => dataset.shuffle(seed),
    next: async () => {
      const sample = await dataset.next();
      if (sample) return transform(sample);
      else return undefined;
    },
    nextBatch: async (batchSize: number) => {
      const samples: Array<OUT> = [];
      while (batchSize--) {
        const s = await transformedData.next();
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
