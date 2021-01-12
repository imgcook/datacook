const tf = require("@tensorflow/tfjs-core");


describe("Encodings", function () {

  describe("LabelEncoder", function () {

    it("test the label encoding on array", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let encode = new datacook.Encoder.LabelEncoder(data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(["dog", "man"]), [0, 2]);
    });


    it("test the label encoding on Tensor", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let tensor_data = tf.tensor(data);
      let encode = new datacook.Encoder.LabelEncoder(tensor_data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(encode.fit(), fit_data);
      assert.deepEqual(encode.transform(tf.tensor(["dog", "man"])), [0, 2]);
    });

  });


  describe("OneHotEncoder", function () {

    it("test onehotencoding on array", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let encode = new datacook.Encoder.OneHotEncoder(data);
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

    it("test onehotencoding on Tensor", function () {
      let data = ["dog", "cat", "man", "dog", "cat", "man", "man", "cat"];
      let tensor_data = tf.tensor(data);
      let encode = new datacook.Encoder.OneHotEncoder(tensor_data);
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
      assert.deepEqual(encode.transform(tf.tensor(["man", "cat"])), transform_data);
    });
  });
});