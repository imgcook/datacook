import { Matrix } from "../classes";
import { transpose2dForward } from "./transform";

export const matMulForward = (x: Matrix, y: Matrix): Matrix => {
  const [ nX, mX ] = x.shape;
  const [ nY, mY ] = y.shape;
  if (mX !== nY) {
    throw new TypeError(`Matrix shape [ ${nX}, ${mX} ] and [ ${nY}, ${mY} ] do not match`);
  }
  const out: number[][] = new Array(nX);
  for (let i = 0; i < nX; i++) {
    out[i] = new Array(mY).fill(0);
  }
  for (let i = 0; i < nX; i++) {
    for (let j = 0; j < mY; j++) {
      for (let k = 0; k < mX; k++) {
        out[i][j] += x.get(i, k) * y.get(k, j);
      }
    }
  }
  return new Matrix(out);
};

export const matMul2d = (x: Matrix, y: Matrix): Matrix => {
  const outMat = matMulForward(x, y);
  outMat.dependency.push({
    target: x,
    gradFunc: (grad: Matrix): Matrix => matMulForward(grad, transpose2dForward(y))
  }, {
    target: y,
    gradFunc: (grad: Matrix): Matrix => matMulForward(transpose2dForward(x), grad)
  });
  return outMat;
};
