import { ENV, TensorContainer } from '@tensorflow/tfjs-core';

/**
 * FIXME(Yorkie): is there a better way to access tfengine instance from tfjs-core?
 */
const engine = ENV.global._tfengine;

/**
 * The tidy-like method that supports Promise, we still recommend you to use tidy() directly and
 * avoid asynchronous operations.
 *
 * See https://github.com/tensorflow/tfjs/commit/5acd02003c3a00256f495ced52a0c1c85bde52fc.
 */
export async function tidyAsync<T extends TensorContainer>(fn: () => Promise<T>): Promise<T> {
  engine.startScope();
  let result: T;
  const end = () => engine.endScope(result);

  try {
    result = await fn();
    end();
    return result;
  } catch (err) {
    end();
    throw err;
  }
}

