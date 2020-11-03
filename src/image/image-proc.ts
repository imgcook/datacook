import Jimp from 'jimp';
import {img2array,tensor2Img,Tensor2ImgArgs,is_img} from './utils';
import { Tensor } from '@tensorflow/tfjs-core';

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
        
    // is_img(name); // check the image extension
    const jimp_obj = await Jimp.read(name);
  
    return new Image(jimp_obj);
    }

  /**
   * Convert image Buffer to Tensor
   * @param depth -> number of channel to obtain from image
   * @return Promise<Array<Array<Array<number>>>> -> A 3dimensional array
   */
  public img2tensor(depth: number=3): Array<Array<Array<number>>>{

    const tensor = img2array({
      data: this.data,
      height: this.height,
      width: this.width,
      channel: 4,
      new_channel: depth,
    });
  
    return tensor
  } 

  /**
   * Convert Image Tensor to one dimensional Array to be converted to Buffer.
   * @param opt {data: number[][][], height:number, width:number, channel:number}
   * @returns img_array number[]
   */
  static tensor2image(opt: Tensor2ImgArgs): Array<number>{
    const img_array = tensor2Img(opt);
    return img_array;
  }
  
  public resize(width:number, height:number): Image {
    this.img = this.img.resize(width,height);
  
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

}
