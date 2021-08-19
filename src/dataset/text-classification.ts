import { Dataset, TextClassification } from './types';
import { ArrayDatasetImpl } from './';

/**
 * Make dataset for text classification.
 * @param textList The list of category and sample text.
 * @returns Dataset for text classification.
 */
export function makeTextClassificationDatasetFromList(textList: TextClassification.TextList): Dataset<TextClassification.Sample> {
  const data = textList.map((textDesc) => {
    return {
      data: textDesc.text,
      label: textDesc.category
    };
  });
  return new ArrayDatasetImpl<TextClassification.Sample>(data);
}
