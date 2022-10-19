export type CSR_MATRIX = { values: number[], rows: number[], cols: number[], n: number, m: number };
export const convertMat2Csr = (mat: number[][]): CSR_MATRIX => {
  const n = mat.length;
  const m = mat[0].length;
  const cols: number[] = [];
  const rows: number[] = [];
  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    let startIdx = -1;
    for (let j = 0; j < m; j++) {
      if (mat[i][j] !== 0) {
        if (startIdx === -1) {
          startIdx = values.length;
        }
        cols.push(j);
        values.push(mat[i][j]);
      }
    }
    rows.push(startIdx);
  }
  return { values, rows, cols, n, m };
};

export const convertCsr2Mat = (mat: CSR_MATRIX): number[][] => {
  const { values, rows, cols, m, n } = mat;
  const outMat: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const arr = new Array(m).fill(0);
    const startIdx = rows[i];
    if (startIdx === -1) {
      outMat[i] = arr;
      continue;
    }
    const endIdx = i < n - 1 ? rows[i + 1] : values.length;
    for (let j = startIdx; j < endIdx; j++) {
      arr[cols[j]] = values[j];
    }
    outMat[i] = arr;
  }
  return outMat;
};

