/**
 * Select kth element in an numeric array
 * @param arr input numeric array
 * @param k index
 */
export const quickSelect = (arr: number[], k: number, pivotFunction: (arr: number[]) => number): number => {
  const n = arr.length;
  if (k < 0 || k >= n) {
    throw new RangeError('Index out of range');
  }
  if (arr.length === 1) {
    return arr[0];
  }
  const pivot = pivotFunction(arr);
  const lows = arr.filter((val: number) => val < pivot);
  const highs = arr.filter((val: number) => val > pivot);
  const pivots = arr.filter((val: number) => val === pivot);
  if (k < lows.length) {
    return quickSelect(lows, k, pivotFunction);
  } else {
    if (k < lows.length + pivots.length) {
      return pivots[0];
    } else {
      return quickSelect(highs, k - lows.length - pivots.length, pivotFunction);
    }
  }
};

/**
 * Choose a number in an array randomly
 * @param arr input array
 * @returns random select numer in arr
 */
export const randomChoice = (arr: number[]): number => {
  const k = Math.floor(Math.random() * arr.length);
  return arr[k];
};

/**
 * Select quantile k for an given array
 * @param arr input array
 * @param quantile quantile
 * @param pivotFunction function for get pivot
 */
export const quickSelectQuantile = (arr: number[], quantile: number, pivotFunction = randomChoice): number => {
  if (quantile === undefined) {
    throw new TypeError('quantile should be provided as a numeric valaue or array');
  }
  if (quantile < 0 || quantile > 1) {
    throw new RangeError('Quantile should be in the range of [0, 1]');
  }
  if (quantile === 0 || quantile * arr.length < 1) {
    return Math.min(...arr);
  }
  if (quantile === 1) {
    return Math.max(...arr);
  }
  const n = arr.length;
  const k = (n - 1) * quantile;
  if (Math.floor(k) == Math.ceil(k)) {
    return quickSelect(arr, k, pivotFunction);
  }
  const r = k - Math.floor(k);
  return quickSelect(arr, Math.floor(k), pivotFunction) * (1 - r) + quickSelect(arr, Math.ceil(k), pivotFunction) * r;
};


export const quickSelectQuantiles = (arr: number[], quantile: number | number[]): number[] | number => {
  if (quantile instanceof Array) {
    const quantileVals = [];
    for (let i = 0; i < quantile.length; i++) {
      quantileVals.push(quickSelectQuantile(arr, quantile[i]));
    }
    return quantileVals;
  } else {
    return quickSelectQuantile(arr, quantile);
  }
};

/**
 * Select median value for an given array
 * @param arr input array
 * @param pivotFunction function for get pivot
 * @returns median value
 */
export const quickSelectMedian = (arr: number[], pivotFunction = randomChoice): number => {
  const n = arr.length;
  if (n % 2 === 1) {
    return quickSelect(arr, n / 2, pivotFunction);
  } else {
    return 0.5 * (quickSelect(arr, Math.floor(n / 2) - 1, pivotFunction) + quickSelect(arr, Math.floor(n / 2), pivotFunction));
  }
};
