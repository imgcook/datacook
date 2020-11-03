import { assert } from "chai"
import Image from "../../src/image/image-proc";
import * as fs from 'fs';

describe("Image-proc", ()=>{

  it("should resize image properly", async ()=>{

    const img: Image = await Image.read("artifacts/dog.jpg")

    const resize_data = [...img.resize(100,100).data];
    const expected_data = [...fs.readFileSync("artifacts/resize100.ds")];
    
    assert.deepEqual(resize_data,expected_data);
  });
});