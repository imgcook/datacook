export type Image = {
  id: number;
  width: number;
  height: number;
  file_name: string;
  license?: number;
  flickr_url?: string;
  coco_url?: string;
  url?: string;
  date_captured?: string;
};

export type Info = {
  year: number;
  version: string;
  description: string;
  contributor: string;
  url: string;
  date_created: string;
};

export type Bbox = [ x: number, y: number, width: number, height: number ];

export type Annotation = {
  id: number;
  image_id: number;
  category_id: number;
  segmentation?: Array<any>;
  area?: number;
  bbox: Bbox;
  iscrowd: 0 | 1;
};

export type Category = {
  supercategory: string;
  id: number;
  name: string;
};

export type Meta = {
  info?: Info;
  licenses?: Array<License>;
  images: Array<Image>;
  annotations: Array<Annotation>;
  categories: Array<Category>;
};

export type License = {
  id: number;
  name: string;
  url: string;
};

export type Label = Array<Annotation>;
