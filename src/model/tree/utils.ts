export const sort = () => {
  
};

/**
 * Median of three pivot selection, after Bentley and McIlroy (1993).
 * @param values 
 * @param n 
 */
export const median3 = (values: number[], start: number, end: number): number => {
  const n = end - start;
  let a = values[start];
  let b = values[start + Math.floor(n / 2)];
  let c = values[end];
  if (a < b) {
    if (b < c) {
      return b;
    }
    if (a < c) {
      return c;
    }
    return a;
  }
  if (b < c) {
    if (a < c) {
      return a;
    }
    return c;
  }
  return b;
};

export const introSort = (values: number[], indicies: number[], start: number, end: number, maxDepth: number): void => {
  let pivot: number;
  let i = 0;
  let l = 0;
  let r = 0;
  let n = end - start;
  while (n > 1) {
    /**
     * max depth limit exceeded ("gone quadratic")
     */
    if (maxDepth <= 0) {
      heapSort(values, indicies, start, end);
      return;
    }
    maxDepth -= 1
    pivot = median3(values, start, end);
    // tree-way partition
    i = l = start;
    r = end;
    while (i < r) {
      if (values[i] < pivot) {
        swap(values, indicies, i, l);
        i += 1;
        l += 1;
      } else {
        if (values[i] > pivot) {
          r -= 1;
          swap(values, indicies, i, r);
        } else {
          i += 1;
        }
      }
    }
    introSort(values, indicies, start, l, maxDepth);
    start = r;
  }
};

export const swap = (values: number[], indices: number[], i: number, j: number): void => {
  if (i === j) return;
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

export const heapSort = (values: number[], indices: number[], startInd: number, endInd: number) => {
  const n = endInd - startInd;
  let start = startInd + (n - 2) / 2;
  let end = endInd;
  // heapify
  while (true) {
    siftDown(values, indices, start, end);
    if (start === 0) {
      break;
    }
    start -= 1;
  }
  end = endInd - 1;
  while (end > start) {
    swap(values, indices, start, end);
    siftDown(values, indices, start , end);
    end = end - 1;
  }
};
