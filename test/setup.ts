import '@tensorflow/tfjs-backend-cpu';
import { memory } from '@tensorflow/tfjs-core';

process.on('exit', function() {
  /**
   * detect the memory leaks by using `tf.memory()`.
   */
  const tfheap = memory();
  if (tfheap.numTensors > 0 || tfheap.numBytes > 0) {
    console.error('memory leaks detected, because the tfjs memroy not get updated');
    console.error(tfheap);
    process.exit(1);
  }
});

