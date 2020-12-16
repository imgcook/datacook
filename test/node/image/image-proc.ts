import { assert, expect } from "chai"
import Image from "../../../src/image/image-proc";
import {stdCalc} from "../../../src/image/utils";
import * as fs from 'fs';
import '@tensorflow/tfjs-backend-cpu';

describe("Image-proc", ()=>{

  it("should resize image properly", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg")
    const resizeData = [...img.resize(100,100).data];
    const expectedData = [...fs.readFileSync("test/node/image/artifacts/resize100.ds")];
    assert.deepEqual(resizeData,expectedData);
  });
  it("should convert image to tensor", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg");
    const resizeTensor = img.resize(50,50).toTensor();
    const expectedTensor = new Int32Array([...fs.readFileSync("test/node/image/artifacts/resize50_tensor.ds")]); 
    assert.deepEqual(resizeTensor.dataSync(),expectedTensor);
  });
  it("should convert tensor3D back to image", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg");

    const resizeTensor = img.resize(50,50).toTensor();
    const resizedArray  = await Image.fromTensor(resizeTensor);

    const resizeData = [...resizedArray.data];
    const expectedData = [...fs.readFileSync("test/node/image/artifacts/tensor2img50.ds")];

    assert.deepEqual(resizeData,expectedData);
  });
  it("should write image to a file", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg"); 
    const resizeImg = img.resize(50,50);
    const isSave = resizeImg.save("test/node/image/artifacts/img_resize.jpg")

    fs.unlinkSync("test/node/image/artifacts/img_resize.jpg");
    expect(isSave).to.be.true;
  });

  it("should convert all image procesing to tensor", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg"); 

    const imgTensor = img.resize(300,300)
                          .crop(0,0,200,200)
                          .flip(true,false)
                          .rotate(90)
                          .toTensor()

    const expectedTensor = new Int32Array([...fs.readFileSync("test/node/image/artifacts/img_chain_tensor.ds")]);
    assert.deepEqual(imgTensor.dataSync(),expectedTensor);
  });

  it("should normalize and unormalize an image tensor", async ()=>{

    const img: Image = await Image.read("test/node/image/artifacts/dog.jpg");

    const resizeTensor = img.resize(50,50).toTensor();
    const img2: Image = await Image.fromTensor(resizeTensor);
    const mean = resizeTensor.mean().round().arraySync() as number;
    const std = stdCalc(resizeTensor);
    
    const normalize = Image.normalize(resizeTensor);
    const unnormalize = Image.unnormalize(normalize,mean,std)
    assert.deepEqual(unnormalize.arraySync(),resizeTensor.arraySync())

  });

});