import { Dataset, Sample } from "./types";
import { range, shuffle } from "../generic";

/**
 * Dataset for array of sample.
 */
export class ArrayDatasetImpl<T extends Sample> implements Dataset<T> {
  private data: Array<T>;
  private dataIndexes: Array<number>;
  private cursor: number;

  constructor(data: Array<T>) {
    this.data = data;
    this.cursor = 0;
    this.dataIndexes = range(0, data.length);
  }

  /**
   * Shuffle data.
   */
  shuffle(): void {
    if (this.data.length === 0) this.dataIndexes = [];
    else shuffle(this.dataIndexes);
  }

  /**
   * Fetch one sample, return null if EOF.
   * @returns one sample or null
   */
  async next(): Promise<T | null> {
    return this.data[this.dataIndexes[this.cursor++]];
  }
  /**
   * Fetch samples in batch, return empty array if EOF.
   * @param batchSize batch size, if not positive integer, fetch all
   * @returns Array of samples
   */
  async nextBatch(batchSize?: number): Promise<Array<T>> {
    const ret: Array<T> = [];

    // return the rest of dataset if non-positive integer present
    if (batchSize <= 0 || batchSize === undefined) {
      let value = await this.next();
      while (value) {
        ret.push(value);
        value = await this.next();
      }
      return ret;
    }

    // default behaviour
    while (batchSize--) {
      const value = await this.next();
      if (!value) break;
      ret.push(value);
    }
    return ret;
  }
  /**
   * Seek cursor to offset.
   * @param offset if small then zero, the cursor will be set to 0, if larger then data length, set to data length.
   */
  async seek(offset: number): Promise<void> {
    if (offset < 0) {
      this.cursor = 0;
    } else if (offset >= this.data.length) {
      this.cursor = this.data.length;
    } else {
      this.cursor = offset;
    }
  }
}

/**
 * Make a transform dataset.
 * @param dataset Origin dataset.
 * @param transform Transform function.
 * @returns The transformed dataset.
 */
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
