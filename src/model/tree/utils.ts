/**
 * Median of three pivot selection, after Bentley and McIlroy (1993).
 * @param values input array
 * @param start start position
 * @param end end position
 */
export const median3 = (values: number[], start: number, end: number): number => {
  const n = end - start;
  const a = values[start],
    b = values[start + Math.floor(n / 2)],
    c = values[end];
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

export const swap = (values: number[], indices: number[], i: number, j: number): void => {
  if (i === j) return;
  [ values[i], values[j] ] = [ values[j], values[i] ];
  [ indices[i], indices[j] ] = [ indices[j], indices[i] ];
};

/**
 * Restore heap order in values[start:end] by moving the max element to start.
 * @param values input array
 * @param indices indices
 * @param start start index
 * @param end end index
 */
export const siftDown = (values: number[], indices: number[], start: number, end: number): void => {
  let child = 0;
  let root = start;
  let maxInd = start;
  while (maxInd !== root) {
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

/**
 * heap sort
 * @param values values
 * @param indices indicies
 * @param startInd start index
 * @param endInd end index
 */
export const heapSort = (values: number[], indices: number[], startInd: number, endInd: number): void => {
  const n = endInd - startInd;
  let start = Math.floor(startInd + (n - 2) / 2);
  let end = endInd;
  // heapify
  while (start >= startInd) {
    siftDown(values, indices, start, end);
    start -= 1;
  }
  end = endInd - 1;
  while (end > start) {
    swap(values, indices, start, end);
    siftDown(values, indices, start, end);
    end = end - 1;
  }
};

/**
 * intro sort
 * @param values values
 * @param indicies indicies
 * @param start start index
 * @param end end index
 * @param maxDepth max depth for quick
 */
export const introSort = (values: number[], indicies: number[], start: number, end: number, maxDepth: number): void => {
  let pivot: number;
  let i = 0;
  let l = 0;
  let r = 0;
  const n = end - start;
  while (n > 1) {
    /**
     * max depth limit exceeded ("gone quadratic")
     */
    if (maxDepth <= 0) {
      heapSort(values, indicies, start, end);
      return;
    }
    maxDepth -= 1;
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

/**
 * Sort n-element arrays pointed to by Xf and samples, simultaneously,
 * @param values values
 * @param indicies input indicies
 * @param start start index for sort
 * @param end end index for sort
 */
export const sort = (values: number[], indicies: number[], start: number, end: number): void => {
  if (start === end) {
    return;
  }
  const n = end - start;
  const maxDepth = 2 * Math.log(n);
  introSort(values, indicies, start, end, maxDepth);
};
