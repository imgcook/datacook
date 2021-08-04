import { Dataset, ImageClassification } from './types';
import { ArrayDatasetImpl } from './';

export function makeClassificationDatasetFromList(imageList: ImageClassification.ImageList): Dataset<ImageClassification.Sample> {
  const data = imageList.map(img => {
    return {
      data: {
        uri: img.uri,
        buffer: img.buffer
      },
      label: img.category
    };
  });
  return new ArrayDatasetImpl<ImageClassification.Sample>(data);
}
