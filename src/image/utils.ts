import * as tf from '@tensorflow/tfjs-core';

interface ImageProp {
  height: number;
  width: number;
  channel: number;

}

export interface Tensor2ImgArgs extends ImageProp{
  data: tf.Tensor3D;
}

interface Img2TensorArgs extends ImageProp{
  data: Buffer;
  new_channel: number;
}

export function img2array(opt: Img2TensorArgs): tf.Tensor3D {

  let img_array: Array<Array<Array<number>>> = [];

  for (let ch = 0; ch < opt.new_channel;ch++){
    let channel_array: Array<Array<number>> = [];

    for (let j = 0; j < opt.width; j++){
      let inner_array: Array<number> = [];

      for (let k = 0; k < opt.height; k++){
        let index: number = ((opt.width * k) + j) * opt.channel + ch;
        inner_array.push(opt.data[index]);
      }
      channel_array.push(inner_array);
    }
    img_array.push(channel_array);
  }
  return tf.tensor3d(img_array);
}

export function tensor2Img(opt: Tensor2ImgArgs): Array<number>{

  let new_array: Array<number> = new Array(opt.width *
      opt.height * opt.channel);

  let data = opt.data.arraySync();
  let channel_array: Array<Array<number>> = [];

  for (let ch = 0; ch < opt.channel; ch++){
    channel_array = data[ch];

    for (let j = 0; j < opt.width; j++){
      for (let k = 0; k < opt.height; k++){
        let index: number = ((opt.width * k) + j) * opt.channel + ch;
        new_array[index] = channel_array[j][k];
      }
    }
  }
  return new_array;
}

export function std(data: tf.Tensor3D): tf.Tensor {

  let tensor_data = data;

  let mean = tensor_data.mean();
  let sub_mean_pow = tensor_data.sub(mean).pow(2);
  let mean_data = sub_mean_pow.mean();
  let std = mean_data.sqrt();

  return std;
}
