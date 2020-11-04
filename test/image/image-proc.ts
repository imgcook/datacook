import { assert, expect } from "chai"
import Image from "../../src/image/image-proc";
import {std} from "../../src/image/utils";
import * as fs from 'fs';
import '@tensorflow/tfjs-backend-cpu';
import { norm } from "@tensorflow/tfjs-core";

describe("Image-proc", ()=>{

  it("should resize image properly", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg")

    const resize_data = [...img.resize(100,100).data];
    const expected_data = [...fs.readFileSync("test/image/artifacts/resize100.ds")];
    
    assert.deepEqual(resize_data,expected_data);
  });
  it("should convert image to tensor", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg");

    const resize_tensor = img.resize(50,50).img2tensor();
    const expected_tensor = new Float32Array([...fs.readFileSync("test/image/artifacts/resize50_tensor.ds")]);
    
    assert.deepEqual(resize_tensor.dataSync(),expected_tensor);
  });
  it("should convert tensor3D back to image", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg");

    const resize_tensor = img.resize(50,50).img2tensor();
    const resize_array  = await Image.tensor2image({
      data: resize_tensor,
      height: img.height,
      width: img.width,
      channel: 3
    });

    const resize_data = [...resize_array.data];
    const expected_data = [...fs.readFileSync("test/image/artifacts/tensor2img50.ds")];

    assert.deepEqual(resize_data,expected_data);
  });
  it("should write image to a file", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg"); 
    const resize_img = img.resize(50,50);
    const is_save = resize_img.save_img("test/image/artifacts/img_resize.jpg")

    fs.unlinkSync("test/image/artifacts/img_resize.jpg");
    expect(is_save).to.be.true;
  });

  it("should convert all image procesing to tensor", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg"); 

    const img_tensor = img.resize(300,300)
                          .crop(0,0,200,200)
                          .flip(true,false)
                          .rotate(90)
                          .img2tensor()

    const expected_tensor = new Float32Array([...fs.readFileSync("test/image/artifacts/img_chain_tensor.ds")]);
    assert.deepEqual(img_tensor.dataSync(),expected_tensor);
  });

  it("should normalize and unormalize an image tensor", async ()=>{

    const img: Image = await Image.read("test/image/artifacts/dog.jpg");

    const resize_tensor = img.resize(50,50).img2tensor();
    const mean = resize_tensor.mean().round().arraySync() as number;
    const t_std = std(resize_tensor).round().arraySync() as number;
    
    const normalize = Image.normalize(resize_tensor);
    const unnormalize = Image.unnormalize(normalize,mean,t_std)
    assert.deepEqual(unnormalize.arraySync(),resize_tensor.arraySync())

  });

});