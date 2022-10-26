/**
 * Select kth element in an numeric array
 * @param arr input numeric array
 * @param k index
 */
export const quickPartitionNode = (data: number[][], idxArr: number[], startIdx: number, endIdx: number, iMax: number, k: number): void => {
  const n = endIdx - startIdx;
  if (k < 0 || k >= n) {
    throw new RangeError('Index out of range');
  }
  if (n === 1) {
    return;
  }
  const pivotIdx = startIdx + Math.floor(Math.random() * n);
  const startIdxOrigin = idxArr[startIdx];
  idxArr[startIdx] = idxArr[pivotIdx];
  idxArr[pivotIdx] = startIdxOrigin;
  const pivot = data[idxArr[startIdx]][iMax];
  let i = startIdx + 1;
  let highCount = 0;
  while (i < endIdx - highCount) {
    if (data[idxArr[i]][iMax] > pivot) {
      const tmp = idxArr[endIdx - highCount - 1];
      idxArr[endIdx - highCount - 1] = idxArr[i];
      idxArr[i] = tmp;
      highCount++;
    } else {
      i++;
    }
  }
  let equalCount = 0;
  i = startIdx;
  while (i < endIdx - highCount - equalCount) {
    if (data[idxArr[i]][iMax] === pivot) {
      const tmp = idxArr[endIdx - highCount - equalCount - 1];
      idxArr[endIdx - highCount - equalCount - 1] = idxArr[i];
      idxArr[i] = tmp;
      equalCount++;
    } else {
      i++;
    }
  }

  const lowCount = n - equalCount - highCount;
  // const curArr = n.map((d) => curData[iMax]);
  if (k < lowCount) {
    return quickPartitionNode(data, idxArr, startIdx, startIdx + lowCount, iMax, k);
  } else {
    if (k < lowCount + equalCount) {
      return;
    } else {
      return quickPartitionNode(data, idxArr, startIdx + lowCount + equalCount, endIdx, iMax, k - lowCount - equalCount);
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
   * Select median value for an given array
   * @param arr input array
   * @param pivotFunction function for get pivot
   * @returns median value
   */
export const partitionNodeIndices = (data: number[][], idxArr: number[], startIdx: number, endIdx: number, iMax: number): void => {
  const k = Math.floor((endIdx - startIdx) / 2);
  return quickPartitionNode(data, idxArr, startIdx, endIdx, iMax, k);
};

