import { Sample, Dataset, ObjectDetection, Coco, PascalVoc } from './types';
import { makeDatasetFromCoco, extractCategoriesFromCoco } from './format/coco';
import { makeDatasetFromPascalVoc } from './format/pascal-voc';
import { makeTransform } from './';

export function makeObjectDetectionDatasetFromCoco(meta: Coco.Meta): Dataset<ObjectDetection.Sample> {
  const dataset = makeDatasetFromCoco(meta);
  const categories = extractCategoriesFromCoco(meta);
  const categoryFinder: Record<number, Coco.Category> = {};
  categories.forEach((item) => {
    categoryFinder[item.id] = item;
  });
  return makeTransform<Coco.Sample, ObjectDetection.Sample>(dataset, async (sample: Sample<Coco.Image, Coco.Label>): Promise<ObjectDetection.Sample> => {
    const newLabels = sample.label.map((lable) => {
      return {
        name: categoryFinder[lable.category_id]?.name,
        bbox: lable.bbox
      };
    });
    return {
      data: { uri: sample.data.url || sample.data.coco_url || sample.data.flickr_url },
      label: newLabels
    };
  });
}

export function makeObjectDetectionDatasetFromPascalVoc(annotations: Array<PascalVoc.Annotation>): Dataset<ObjectDetection.Sample> {
  const dataset = makeDatasetFromPascalVoc(annotations);
  return makeTransform<PascalVoc.Sample, ObjectDetection.Sample>(
    dataset,
    async (sample: PascalVoc.Sample): Promise<ObjectDetection.Sample> => {
      const newLabels: ObjectDetection.Label = sample.label.map((label) => {
        return {
          name: label.name,
          bbox: [
            label.bndbox.xmin,
            label.bndbox.ymin,
            label.bndbox.xmax - label.bndbox.xmin,
            label.bndbox.ymax - label.bndbox.ymin
          ]
        };
      });
      return {
        data: { uri: sample.data.path },
        label: newLabels
      };
    }
  );
}
