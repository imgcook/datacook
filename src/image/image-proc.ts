import Jimp from 'jimp';
import { img2array, tensor2Img, Tensor2ImgArgs, std } from './utils';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';

/**
 * Image class contains utility to handle image manipulation
 * @contructor (name: string) -> name is the image file name.
 */
export default class Image {
  private img: Jimp; //img accessible to all methods in the class

  constructor (data: Jimp){
    this.img = data;
  }

  /**
   * Read image from files and create an Image object for processing
   * @param name image file name
   */
  static async read(name: string): Promise<Image>{

    const jimp_obj = await Jimp.read(name);

    return new Image(jimp_obj);
  }

  /**
   * Convert image Buffer to Tensor
   * @param depth -> number of channel to obtain from image
   * @return Promise<Array<Array<Array<number>>>> -> A 3dimensional array
   */
  public img2tensor(depth = 3): tf.Tensor3D{

    const tensor = img2array({
      data: this.data,
      height: this.height,
      width: this.width,
      channel: 4,
      new_channel: depth
    });

    return tensor;
  }

  /**
   * Convert Image Tensor to one dimensional Array to be converted to Buffer.
   * @param opt {data: number[][][], height:number, width:number, channel:number}
   * @returns img_array number[]
   */
  static async tensor2image(opt: Tensor2ImgArgs): Promise<Image>{
    const img_array = tensor2Img(opt);
    const data: Buffer = Buffer.from(img_array);

    const img: Jimp = await new Jimp({ data:data, width:opt.width, height:opt.height });
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

  get width(): number{
    return this.img.bitmap.width;
  }

  get height(): number{
    return this.img.bitmap.height;
  }

  get data(): Buffer{
    return this.img.bitmap.data;
  }

  /**
   * save the image into a file
   * @param name the image file name
   * @return Boolean
   */
  public save_img(name:string): boolean{

    const is_save = this.img.write(name);
    if (is_save){
      return true;
    }
    return false;
  }

  /**
   * Rotate image
   * @param deg rotation degree
   * @return Image
   */
  public rotate(deg:number): Image{
    this.img = this.img.rotate(deg);
    return this;
  }

  /**
   * flip the image to horizontal or vertical
   * @param horz
   * @param vert
   * @return Image
   */
  public flip(horz: boolean, vert: boolean): Image{
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
  public crop(x:number, y:number, width:number, height:number): Image{
    this.img = this.img.crop(x, y, width, height);
    return this;
  }

  /**
   * Normalize images to [-1,1] using (tensor - mean) / std
   * @param data
   * @param mean
   * @param im_std
   * @return Tensor3D
   */
  static normalize(data: tf.Tensor3D, mean?: number, im_std?:number): tf.Tensor3D{

    const tf_mean = mean ? mean : data.mean().round().arraySync();
    const tf_std = im_std ? im_std : std(data).round().arraySync();
    const norm = data.sub(tf_mean).div(tf_std);

    return norm as tf.Tensor3D;
  }

  /**
   * Un-normalize a normalize image tensor
   * @param data
   * @param mean
   * @param tf_std
   * @returns Tensor
   */
  static unnormalize(data: tf.Tensor3D, mean: number, tf_std:number): tf.Tensor3D{
    const unnorm = tf.cast(data.mul(tf_std).add(mean), "int32");
    return unnorm as tf.Tensor3D;
  }

}
