/**
 * matmul of two 2D matrices of shape (m, n) and (n, k)
 * @param a matrix of shape (m, n)
 * @param b matrix of shape (n, k)
 * @returns a * b
 */
export const arrayMatmul2D = (a: number[][], b: number[][]): number[][] => {
  const aNumRows = a.length, aNumCols = a[0].length,
    bNumRows = b.length, bNumCols = b[0].length,
    m = new Array(aNumRows); // initialize array of rows
  if (bNumRows !== aNumCols) {
    throw new TypeError('dimension of two matrices do not match');
  }
  for (let r = 0; r < aNumRows; ++r) {
    m[r] = new Array(bNumCols); // initialize the current row
    for (let c = 0; c < bNumCols; ++c) {
      m[r][c] = 0; // initialize the current cell
      for (let i = 0; i < aNumCols; ++i) {
        m[r][c] += a[r][i] * b[i][c];
      }
    }
  }
  return m;
};

/**
 * Generate 2D diag values of an array
 * @param a diag values
 */
export const arrayDiag = (a: number[]): number[][] => {
  const res = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const item = new Array(a.length).fill(0);
    item[i] = a[i];
    res[i] = item;
  }
  return res;
};

/**
 * Transpose of an 2D array
 */
export const arrayTranpose2D = (a: number[][]): number[][] => {
  const m = a.length;
  const n = a[0].length;
  const c = new Array(n);
  for (let i = 0; i < n; i++) {
    c[i] = new Array(m);
  }
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      c[j][i] = a[i][j];
    }
  }
  return c;
};

export const arrayMean1D = (array: number[]): number => {
  return array.reduce((a: number, b: number) => a + b) / array.length;
};

export const arrayVariance1D = (array: number[], mean?: number): number => {
  mean = mean ? mean : arrayMean1D(array);
  const centered = array.map((d: number) => (d - mean) * (d - mean));
  return centered.reduce((a: number, b: number) => a + b) / (array.length - 1);
};
