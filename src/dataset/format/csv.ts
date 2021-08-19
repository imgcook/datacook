import { Dataset, Csv } from '../types';
import { ArrayDatasetImpl } from '../utils';

function toSamples(
  parsedData: Array<Record<string, any>>,
  labelFields?: Array<string>
): Array<Csv.Sample> {
  return parsedData.map((data) => {
    const label: Record<string, string> = {};
    const newData = { ...data };
    labelFields?.forEach((field) => {
      label[field] = newData[field];
      delete newData[field];
    });
    return {
      data: newData,
      label
    };
  });
}

/**
 * Make dataset from csv.
 * @param records The csv records.
 * @param labelKeys The label colume name or index list.
 * @returns Csv dataset.
 */
export const makeDatasetFromCsv = (
  records: Array<Record<string, any>>,
  labelKeys?: Array<string>
): Dataset<Csv.Sample> => {
  const data = toSamples(records, labelKeys);
  return new ArrayDatasetImpl(data);
};
