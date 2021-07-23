import { Sample, ObjectDetection, Coco, PascalVoc } from './types';
import { makeDatasetFromCocoFormat } from './format/coco';
import { transformDataset } from './';
import { makeDatasetFromPascalVocFormat } from './format/pascal-voc';

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

export const makeObjectDetectionDatasetFromPascalVoc = async (options: PascalVoc.Options): Promise<ObjectDetection.Dataset> => {
  const dataset = await makeDatasetFromPascalVocFormat(options);
  return transformDataset<
      PascalVoc.DatasetMeta,
      Sample<PascalVoc.ExtAnnotation, Array<PascalVoc.ExtPascalVocObject>>,
      ObjectDetection.Sample,
      ObjectDetection.DatasetMeta
    >({
      next: async (sample: Sample<PascalVoc.ExtAnnotation, Array<PascalVoc.ExtPascalVocObject>>)
        : Promise<ObjectDetection.Sample> => {
        const newLabels: ObjectDetection.Label = sample.label.map((lable) => {
          return {
            id: lable.id,
            bbox: [
              lable.bndbox.xmin,
              lable.bndbox.ymin,
              lable.bndbox.xmax - lable.bndbox.xmin,
              lable.bndbox.ymax - lable.bndbox.ymin
            ]
          };
        });
        return {
          data: `${sample.data.folder ? sample.data.folder + '/' : ''}${sample.data.filename}`,
          label: newLabels
        };
      },
      metadata: async (meta: PascalVoc.DatasetMeta): Promise<ObjectDetection.DatasetMeta> => {
        const labelMap: Record<number, string> = {};
        for (const labelId in meta.labelMap) {
          labelMap[labelId] = meta.labelMap[labelId];
        }
        return {
          type: meta.type,
          size: meta.size,
          labelMap
        };
      }
    }, dataset);
};
