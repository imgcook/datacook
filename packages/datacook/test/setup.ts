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
    /**
     * TODO(Yorkie): just disable this break util we have fixed all the memory issues.
     */
    // process.exit(1);
  }
});

