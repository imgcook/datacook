export const sort = () => {
  
};

export const introSort = () => {

};

export const swap = (values: number[], indices: number[], i: number, j: number): void => {
  [values[i], values[j]] = [values[j], values[i]];
  [indices[i], indices[j]] = [indices[j], indices[i]];
};

/**
 * Restore heap order in values[start:end] by moving the max element to start.
 * @param values 
 * @param indices 
 * @param start 
 * @param end 
 */
export const siftDown = (values: number[], indices: number[], start: number, end: number) => {
  let child = 0;
  let root = start;
  let maxInd = start;
  while (true) {
    child = root * 2 + 1;
    maxInd = root;
    if (child < end && values[maxInd] < values[child]) {
      maxInd = child;
    }
    if (child + 1 < end && values[maxInd] < values[child + 1]) {
      maxInd = child + 1;
    }
    if (maxInd === root) {
      break;
    } else {
      swap(values, indices, root, maxInd);
      root = maxInd;
    }
  }
};

export const heapSort = (values: number[], indices: number[], n: number) => {
  let start = (n - 2) / 2;
  let end = n;
  // heapify
  while (true) {
    siftDown(values, indices, start, end);
    if (start === 0) {
      break;
    }
    start -= 1;
  }
  end = n - 1
  while (end > 0) {
    swap(values, indices, 0, end);
    siftDown(values, indices, 0 , end);
    end = end - 1;
  }
};
