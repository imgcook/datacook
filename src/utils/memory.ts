import { ENV, TensorContainer } from '@tensorflow/tfjs-core';

/**
 * FIXME(Yorkie): is there a better way to access tfengine instance from tfjs-core?
 */
const engine = ENV.global._tfengine;

/**
 * `tidy()` depends on engine's scoped runner, unfortunately the engine is singleton
 * and it doesn't support named scope, the tidy have to implement it in singleton.
 *
 * TODO(Yorkie): help tfjs to implement named scope, then we could remove this state.
 */
let isAsynchronousTidyRunning = false;

/**
 * The tidy-like method that supports Promise, we still recommend you to use tidy() directly and
 * avoid asynchronous operations. In addition, you need to pay attention to it when you use it,
 * you must ensure that only 1 tidy is running globally.
 *
 * See https://github.com/tensorflow/tfjs/commit/5acd02003c3a00256f495ced52a0c1c85bde52fc.
 */
export async function tidyAsync<T extends TensorContainer>(fn: () => Promise<T>): Promise<T> {
  if (isAsynchronousTidyRunning === true) {
    throw new TypeError('tidy must have only 1 running instance, please create a new tidy after others end.');
  }
  engine.startScope();
  isAsynchronousTidyRunning = true;

  let result: T;
  const end = () => {
    engine.endScope(result);
    isAsynchronousTidyRunning = false;
  };

  try {
    result = await fn();
    end();
    return result;
  } catch (err) {
    end();
    throw err;
  }
}

