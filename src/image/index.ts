import Jimp from 'jimp';
import { stdCalc } from './utils';
import * as tf from '@tensorflow/tfjs-core';

/**
 * Image class contains utility to handle image manipulation
 * @contructor (data: Jimp)
 */
export class Image {
  private img: Jimp; //img accessible to all methods in the class

  constructor (data: Jimp) {
    this.img = data;
  }

  /**
   * Read image from files and create an Image object for processing
   * @param data image file name or image buffer
   */
  static async read(data: string | ArrayBuffer): Promise<Image> {
    let jimpObj;
    if (typeof data === 'string') {
      jimpObj = await Jimp.read(data);
    } else {
      jimpObj = await Jimp.read(Buffer.from(data));
    }
    return new Image(jimpObj);
  }

  /**
   * Convert image Buffer to Tensor
   * @param depth -> number of channel to obtain from image
   * @return tf.Tensor3D
   */
  public toTensor(depth = 3): tf.Tensor3D {
    return tf.browser.fromPixels({
      data: this.data,
      height: this.height,
      width: this.width
    }, depth);

  }

  /**
   * Convert Image Tensor to one dimensional Array to be converted to Buffer.
   * @param tensor Tensor3D of image
   * @returns Promise<Image>
   */
  static async fromTensor(tensor: tf.Tensor3D): Promise<Image> {
    const imgArray = await tf.browser.toPixels(tensor);
    const data: Buffer = Buffer.from(imgArray);
    const [ width, height ] = tensor.shape.slice(0, 2);
    const img: Jimp = await new Jimp({ data:data, width:width, height:height });
    return new Image(img);
  }

  /**
   * Resize image
   * @param width number
   * @param height number
   * @return Image
   */
  public resize(width:number, height:number): Image {
    this.img = this.img.resize(width, height);

    return this;
  }

  get width(): number {
    return this.img.bitmap.width;
  }

  get height(): number {
    return this.img.bitmap.height;
  }

  get channel(): number {
    return this.img.hasAlpha() ? 4 : 3;
  }

  get data(): Buffer {
    return this.img.bitmap.data;
  }

  /**
   * save the image into a file
   * @param name the image file name
   * @return Boolean
   */
  public save(name:string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.img.write(name, (err) => {
        if (err) reject(err);
        resolve(true);
      });
    });
  }

  /**
   * Rotate image
   * @param deg rotation degree
   * @return Image
   */
  public rotate(deg:number): Image {
    this.img = this.img.rotate(deg);
    return this;
  }

  /**
   * flip the image to horizontal or vertical
   * @param horz
   * @param vert
   * @return Image
   */
  public flip(horz: boolean, vert: boolean): Image {
    this.img = this.img.flip(horz, vert);
    return this;
  }

  /**
   * Crop the desired side of an image
   * @param x
   * @param y
   * @param width
   * @param height
   * @return Image
   */
  public crop(x:number, y:number, width:number, height:number): Image {
    this.img = this.img.crop(x, y, width, height);
    return this;
  }

  /**
   * Normalize images to [-1,1] using (tensor - mean) / std
   * @param data
   * @param mean
   * @param std
   * @return Tensor3D
   */
  static normalize(data: tf.Tensor3D, mean?: number, std?:number): tf.Tensor3D {
    const tfMean = mean ? mean : tf.round(tf.mean(data)).arraySync();
    const tfStd = std ? std : stdCalc(data);
    const norm = tf.div(tf.sub(data, tfMean), tfStd);
    return norm as tf.Tensor3D;
  }

  /**
   * Un-normalize a normalize image tensor
   * @param data
   * @param mean
   * @param std
   * @returns Tensor
   */
  static unnormalize(data: tf.Tensor3D, mean: number, std:number): tf.Tensor3D {
    const unnorm = tf.cast(tf.add(tf.mul(data, std), mean), 'int32');
    return unnorm as tf.Tensor3D;
  }

}
