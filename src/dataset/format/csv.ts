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

export const makeDatasetFromCsv = (
  records: Array<Record<string, any>>,
  labelKeys?: Array<string>
): Dataset<Csv.Sample> | undefined => {
  if (!records) {
    return undefined;
  }
  const data = toSamples(records, labelKeys);
  return new ArrayDatasetImpl(data);
};
