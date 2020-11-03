interface ImageProp {
  height: number;
  width: number;
  channel: number;

}

export interface Tensor2ImgArgs extends ImageProp{
  data: Array<Array<Array<number>>>;
}

interface Img2TensorArgs extends ImageProp{
  data: Buffer;
  new_channel: number;
}

export function img2array(opt: Img2TensorArgs): Array<Array<Array<number>>> {

  let img_array: Array<Array<Array<number>>> = [];

  for(let ch=0; ch < opt.new_channel;ch++){
    let channel_array: Array<Array<number>> = [];

    for(let j=0; j < opt.width; j++){
      let inner_array: Array<number> = [];

      for(let k=0; k < opt.height; k++){
        let index: number = ((opt.width*k)+j)*opt.channel +ch;
        inner_array.push(opt.data[index]);
      }
      channel_array.push(inner_array);
    }
    img_array.push(channel_array);
  }
  return img_array;
}

export function tensor2Img(opt: Tensor2ImgArgs): Array<number>{

  let new_array: Array<number> = new Array(opt.width*
      opt.height*opt.channel);
  
  let channel_array: Array<Array<number>> = [];

  for(let ch=0; ch < opt.channel; ch++){
    channel_array = opt.data[ch];

    for(let j=0; j < opt.width; j++){
        for(let k=0; k< opt.height; k++){
          let index: number  = ((opt.width*k)+j)*opt.channel +ch;
          new_array[index] = channel_array[j][k];
        }
    }
  }
  return new_array;
}

export function is_img(img: string){
    
  let ext: string = img.split(".")[1];
  let img_ext = ["png", "jpg","JPEG","PNG"];
  
  if(!img_ext.includes(ext)){
      throw new Error(`extension ${ext} not supported`);
  }
}
