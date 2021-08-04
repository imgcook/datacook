import { Dataset, TextClassification } from './types';
import { ArrayDatasetImpl } from './';

export function makeTextClassificationDatasetFromList(textList: TextClassification.TextList): Dataset<TextClassification.Sample> {
  const data = textList.map(textDesc => {
    return {
      data: textDesc.text,
      label: textDesc.category
    };
  });
  return new ArrayDatasetImpl<TextClassification.Sample>(data);
}
