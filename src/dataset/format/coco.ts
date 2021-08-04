import { Dataset, Coco } from '../types';
import { ArrayDatasetImpl } from '../utils';

function checkCocoMeta(metaObj: Record<string, any>) {
  if (!Array.isArray(metaObj.images)) {
    throw new TypeError('images should be array');
  }
  if (!Array.isArray(metaObj.annotations)) {
    throw new TypeError('annotations should be array');
  }
  metaObj.images.forEach((image) => {
    if (typeof image.id !== 'number') {
      throw new TypeError('invalid id field found in image data');
    }
    if (typeof image.width !== 'number') {
      throw new TypeError('invalid width field found in image data');
    }
    if (typeof image.height !== 'number') {
      throw new TypeError('invalid height field found in image data');
    }
    if (
      typeof image.url !== 'string'
      && typeof image.coco_url !== 'string'
      && typeof image.flickr_url !== 'string'
    ) {
      throw new TypeError('invalid url/flickr_url/coco_url field found in image data');
    }
  });
}

function cocoMetaToSamples(cocoMeta: Coco.Meta): Array<Coco.Sample> {
  const annotationMap: Record<number, Array<Coco.Annotation>> = {};
  for (const ann of cocoMeta.annotations) {
    if (!annotationMap[ann.image_id]) {
      annotationMap[ann.image_id] = [];
    }
    annotationMap[ann.image_id].push(ann);
  }
  return cocoMeta.images.map((img: Coco.Image) => ({ data: img, label: annotationMap[img.id] }));
}

export function makeDatasetFromCoco(meta: Coco.Meta): Dataset<Coco.Sample> {
  checkCocoMeta(meta);
  return new ArrayDatasetImpl(cocoMetaToSamples(meta));
}

export function extractCategoriesFromCoco(meta: Coco.Meta): Array<Coco.Category> {
  return meta.categories;
}
