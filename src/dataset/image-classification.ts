import { Dataset, ImageClassification } from './types';
import { ArrayDatasetImpl } from './';

/**
 * Make dataset for image classification from image list.
 * @param imageList Array of image description.
 * @returns Dataset for image classification.
 */
export function makeImageClassificationDatasetFromList(imageList: ImageClassification.ImageList): Dataset<ImageClassification.Sample> {
  const data = imageList.map((img) => {
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
