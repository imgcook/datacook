import { ObjectDetection, Coco, Sample } from './types';
import { makeDatasetFromCocoFormat } from './format/coco';
import { transformDataset } from './';

export const makeObjectDetectionDatasetFromCoco = async (options: Coco.Options): Promise<ObjectDetection.Dataset> => {
  const dataset = await makeDatasetFromCocoFormat(options);
  return transformDataset<Coco.DatasetMeta, Sample<Coco.Image, Coco.Label>, ObjectDetection.Sample, ObjectDetection.DatasetMeta>({
    next: async (sample: Sample<Coco.Image, Coco.Label>): Promise<ObjectDetection.Sample> => {
      const newLabels = sample.label.map((lable) => {
        return {
          id: lable.id,
          bbox: lable.bbox
        };
      });
      return {
        data: sample.data.url || sample.data.coco_url || sample.data.flickr_url,
        label: newLabels
      };
    },
    metadata: async (meta: Coco.DatasetMeta): Promise<ObjectDetection.DatasetMeta> => {
      const labelMap: Record<number, string> = {};
      for (const labelId in meta.labelMap) {
        labelMap[labelId] = meta.labelMap[labelId].name;
      }
      return {
        type: meta.type,
        size: meta.size,
        labelMap
      };
    }
  }, dataset);
};
