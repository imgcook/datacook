import { LabelEncoder, OneHotEncoder } from '../../../src/tabular/encoder';
import { Tensor, tensor1d } from "@tensorflow/tfjs-core";
import { assert } from 'chai';
import { Series } from "danfojs-node";
import 'mocha';


describe("Encodings", function () {

  describe("LabelEncoder", function () {

    it("test the label encoding on array", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let encode = new LabelEncoder(data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(["dog", "man"]), [0, 2]);
    });

    it("test the label encoding on Series", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let series = new Series(data);
      let encode = new LabelEncoder(series);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(new Series(["dog", "man"])), [0, 2]);
    });

    it("test the label encoding on Tensor", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let tensor_data = tensor1d(data);
      let encode = new LabelEncoder(tensor_data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(tensor1d(["dog", "man"])), [0, 2]);
    });

  });


  describe("OneHotEncoder", function () {

    it("test onehotencoding on array", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let encode = new OneHotEncoder(data);
      let fit_data = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
      ];
      let transform_data = [[0, 0, 1], [0, 1, 0]];

      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(["man", "cat"]), transform_data);
    });


    it("test onehotencoding on Series", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let series = new Series(data);
      let encode = new OneHotEncoder(series);
      let fit_data = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
      ];
      let transform_data = [[0, 0, 1], [0, 1, 0]];

      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(new Series(["man", "cat"])), transform_data);
    });

    it("test onehotencoding on Tensor", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let tensor_data = tensor1d(data);
      let encode = new OneHotEncoder(tensor_data);
      let fit_data = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 1, 0]
      ];
      let transform_data = [[0, 0, 1], [0, 1, 0]];

      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(tensor1d(["man", "cat"])), transform_data);
    });
  });
});