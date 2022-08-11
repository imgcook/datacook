import { createZeroMatrix, Matrix, Vector } from "../classes";

export const squeeze = (x: Matrix): Vector => {
  let out: number[] = [];
  const nX = x.shape[0];
  for (let i = 0; i < nX; i++) {
    out = [ ...out, ...x.getRow(i).data ];
  }
  return new Vector(out);
};

export const transpose2dForward = (x: Matrix): Matrix => {
  const [ n, m ] = x.shape;
  const mat = createZeroMatrix(m, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      mat.set(j, i, x.get(i, j));
    }
  }
  return mat;
};

export const transpose2dBackward = (grad: Matrix): Matrix => {
  return transpose2dForward(grad);
};

export const transpose2d = (x: Matrix): Matrix => {
  const outMat = transpose2dForward(x);
  outMat.dependency.push({
    target: x,
    gradFunc: transpose2dBackward
  });
  return outMat;
};


