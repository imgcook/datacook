require('@tensorflow/tfjs-backend-cpu');
const tf = require('@tensorflow/tfjs-core');

describe('Encodings', function () {
  describe('LabelEncoder', function () {
    it('test the label encoding on array', async function () {
      let data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat'];
      let encode = new datacook.Encoder.LabelEncoder(data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(await encode.fit(data), fit_data);
      assert.deepEqual(await encode.transform(['dog', 'man']), [0, 2]);
    });

    it('test the label encoding on Tensor', async function () {
      let data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat'];
      let tensor_data = tf.tensor(data);
      let encode = new datacook.Encoder.LabelEncoder(tensor_data);
      let fit_data = [
        0, 1, 2, 0,
        1, 2, 2, 1
      ];
      assert.deepEqual(await encode.fit(tensor_data), fit_data);
      assert.deepEqual(await encode.transform(tf.tensor(['dog', 'man'])), [0, 2]);
    });
  });


  describe('OneHotEncoder', function () {
    it('test onehotencoding on array', async function () {
      let data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat'];
      let encode = new datacook.Encoder.OneHotEncoder();
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
      assert.deepEqual(await encode.fit(data), fit_data);
      assert.deepEqual(await encode.transform(['man', 'cat']), transform_data);
    });

    it('test onehotencoding on Tensor', async function () {
      let data = ['dog', 'cat', 'man', 'dog', 'cat', 'man', 'man', 'cat'];
      let tensor_data = tf.tensor(data);
      let encode = new datacook.Encoder.OneHotEncoder();
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
      assert.deepEqual(await encode.fit(tensor_data), fit_data);
      assert.deepEqual(await encode.transform(tf.tensor(['man', 'cat'])), transform_data);
    });
  });
});
