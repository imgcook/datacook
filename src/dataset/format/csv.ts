import * as Papaparse from 'papaparse';
import { Dataset, DatasetType, Csv } from '../types';
import { makeDataset } from '../utils';

function toSamples(
  parsedData: Papaparse.ParseResult<Record<string, string>>,
  labelFields?: Array<string>
): Array<Csv.Sample> {
  return parsedData.data.map((data) => {
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

export const makeDatasetFromCsv = async (options: Csv.Options): Promise<Dataset<Csv.Sample, Csv.DatasetMeta>> => {
  const config = {
    header: options.hasHeader, delimiter: options.delimiter
  };
  const parsedTrainData = Papaparse.parse<Record<string, string>>(options.trainData, config);
  const parsedTestData = Papaparse.parse<Record<string, string>>(options.testData, config);
  const parsedValidData = options.validData ? Papaparse.parse<Record<string, string>>(options.validData, config) : undefined;
  const data = {
    trainData: toSamples(parsedTrainData, options.labels),
    testData: toSamples(parsedTestData, options.labels),
    validData: parsedValidData ? toSamples(parsedTestData, options.labels) : undefined
  };
  const meta: Csv.DatasetMeta = {
    type: DatasetType.Table,
    size: {
      train: data.trainData.length,
      test: data.testData.length,
      valid: data.validData ? data.validData.length : 0
    },
    labelMap: undefined
  };
  return makeDataset(data, meta);
};
