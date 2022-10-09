/**
 * Heap structure for saving neighbor indices and distances
 */
export class NeighborHeap {
  public distances: number[][];
  public indices: number[][];
  constructor(nData: number, nNeighbors: number) {
    this.distances = new Array(nData).fill([]).map(() => new Array(nNeighbors).fill(Number.MAX_SAFE_INTEGER));
    this.indices = new Array(nData).fill([]).map(() => new Array(nNeighbors).fill(0));
  }
  public push(row: number, val: number, iVal: number): void {
    const distArr = this.distances[row];
    const indArr = this.indices[row];
    const size = distArr.length;
    if (val > distArr[0]) {
      return;
    }
    distArr[0] = val;
    indArr[0] = iVal;
    // Descend the heap
    let i = 0, iSwap = 0;
    while (i <= size) {
      const ic1 = 2 * i + 1;
      const ic2 = ic1 + 1;
      if (ic1 >= size) {
        break;
      } else {
        if (ic2 >= size) {
          if (distArr[ic1] > val) {
            iSwap = ic1;
          } else {
            break;
          }
        } else {
          if (distArr[ic1] >= distArr[ic2]) {
            if (val < distArr[ic1]) {
              iSwap = ic1;
            } else {
              break;
            }
          } else {
            if (val < distArr[ic2]) {
              iSwap = ic2;
            } else {
              break;
            }
          }
        }
        distArr[i] = distArr[iSwap];
        indArr[i] = indArr[iSwap];
        i = iSwap;
      }
    }
    distArr[i] = val;
    indArr[i] = iVal;
    this.distances[i] = distArr;
    this.indices[i] = indArr;
  }
}
