import { Coco, PascalVoc } from '../../../src/dataset/types';

export const cocoAnnotation: Coco.Meta = {
  images: [
    {
      file_name: 'f984d880-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/f984d880-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 1,
      height: 286
    },
    {
      file_name: 'fb6a8870-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/fb6a8870-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 2,
      height: 363
    },
    {
      file_name: 'fd5abfb0-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      width: 750,
      url: 'img/fd5abfb0-1cb6-11ea-a3c0-69b27346a20f-screenshot.png',
      id: 3,
      height: 286
    }
  ],
  annotations: [
    {
      image_id: 1,
      id: 1,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        36,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 1,
      id: 2,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        270,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 2,
      id: 3,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        270,
        36,
        210,
        250
      ],
      category_id: 1
    },
    {
      image_id: 2,
      id: 4,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        170,
        36,
        110,
        150
      ],
      category_id: 2
    },
    {
      image_id: 3,
      id: 5,
      segmentation: [],
      iscrowd: 0,
      bbox: [
        150,
        136,
        110,
        50
      ],
      category_id: 1
    }
  ],
  categories: [
    {
      supercategory: 'abovePicture',
      id: 1,
      name: 'abovePicture'
    },
    {
      supercategory: 'button',
      id: 2,
      name: 'button'
    }
  ]
};

export const pascalVocAnnotation: Array<PascalVoc.Annotation> = [
  {
    folder: 'images',
    filename: '0001.jpg',
    path: 'images/0001.jpg',
    source: {
      database: 'database',
      annotation: 'source.annotation',
      image: 'source.image',
      flickrid: '123'
    },
    owner: {
      flickrid: 'owner.flickerid',
      name: 'owner.name'
    },
    size: {
      width: 234,
      height: 345,
      depth: 3
    },
    segmented: 0,
    object: [
      {
        // id: 0,
        name: 'dog',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 48,
			    ymin: 240,
			    xmax: 195,
			    ymax: 371
        }
      },
      {
        // id: 1,
        name: 'person',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 8,
			    ymin: 12,
			    xmax: 352,
			    ymax: 498
        }
      }
    ]
  },
  {
    folder: 'images',
    filename: '0002.jpg',
    path: 'images/0002.jpg',
    source: {
      database: 'database',
      annotation: 'source.annotation',
      image: 'source.image',
      flickrid: '123'
    },
    owner: {
      flickrid: 'owner.flickerid',
      name: 'owner.name'
    },
    size: {
      width: 234,
      height: 345,
      depth: 3
    },
    segmented: 0,
    object: [
      {
        // id: 0,
        name: 'dog',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 48,
			    ymin: 240,
			    xmax: 195,
			    ymax: 371
        }
      },
      {
        // id: 1,
        name: 'person',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 8,
			    ymin: 12,
			    xmax: 352,
			    ymax: 498
        }
      }
    ]
  },
  {
    folder: 'images',
    filename: '0003.jpg',
    path: 'images/0003.jpg',
    source: {
      database: 'database',
      annotation: 'source.annotation',
      image: 'source.image',
      flickrid: '123'
    },
    owner: {
      flickrid: 'owner.flickerid',
      name: 'owner.name'
    },
    size: {
      width: 235,
      height: 346,
      depth: 3
    },
    segmented: 0,
    object: [
      {
        // id: 2,
        name: 'dog2',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 48,
			    ymin: 240,
			    xmax: 195,
			    ymax: 371
        }
      },
      {
        // id: 3,
        name: 'person2',
		    pose: 'Left',
		    truncated: 1,
		    difficult: 0,
		    bndbox: {
			    xmin: 8,
			    ymin: 12,
			    xmax: 352,
			    ymax: 498
        }
      }
    ]
  }
];

export const csvDataWithHead = 'A,B,C\n1,2,3\n4,5,6\n7,8,9';
export const csvDataWithoutHead = '1,2,3\n4,5,6\n7,8,9';
