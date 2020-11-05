import * as tf from '@tensorflow/tfjs-core';

interface ImageProp {
  height: number;
  width: number;
  channel: number;

}

interface Img2TensorArgs extends ImageProp {
  data: Buffer;
  newChannel: number;
}

export function img2array(opt: Img2TensorArgs): tf.Tensor3D {

  let imgArray: Array<Array<Array<number>>> = [];

  for (let ch = 0; ch < opt.newChannel;ch++){
    let channelArray: Array<Array<number>> = [];

    for (let j = 0; j < opt.width; j++){
      let innerArray: Array<number> = [];

      for (let k = 0; k < opt.height; k++){
        let index: number = ((opt.width * k) + j) * opt.channel + ch;
        innerArray.push(opt.data[index]);
      }
      channelArray.push(innerArray);
    }
    imgArray.push(channelArray);
  }
  return tf.tensor3d(imgArray);
}

export function tensor2Img(tensor: tf.Tensor3D): Array<number> {

  const [ channel, width, height ] = tensor.shape;
  let newArray: Array<number> = new Array(width * height * channel);

  let data = tensor.arraySync();
  let channelArray: Array<Array<number>> = [];

  for (let ch = 0; ch < channel; ch++){
    channelArray = data[ch];

    for (let j = 0; j < width; j++){
      for (let k = 0; k < height; k++){
        let index: number = ((width * k) + j) * channel + ch;
        newArray[index] = channelArray[j][k];
      }
    }
  }
  return newArray;
}

export function stdCalc(data: tf.Tensor3D): tf.Tensor {

  let tensorData = data;

  let mean = tensorData.mean();
  let subMeanPow = tensorData.sub(mean).pow(2);
  let meanData = subMeanPow.mean();
  let std = meanData.sqrt();

  return std;
}
