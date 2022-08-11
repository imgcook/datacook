import { Vector } from "../backend-cpu/classes";
import { abs2d, max2d, sub2d, abs1d, max1d, sub1d } from "../backend-cpu/op";
import { Matrix, Scalar } from "../core/classes";

export function checkJsArray2D(arr: number[][]): void {
  if (!arr || !arr.length) {
    throw new TypeError('Invalid input, two dimensional array is expected');
  }
  if (!(Array.isArray(arr[0]))) {
    throw new TypeError('Invalid input, two dimensional array is expected');
  }
  const m = arr[0].length;
  for (let i = 0; i < arr.length; i++) {
    if (!(Array.isArray(arr[i])) || arr[i].length !== m) {
      throw new TypeError('Invalid input, array length should be the same on each row');
    }
    for (let j = 0; j < arr[i].length; j++) {
      if (typeof arr[i][j] !== 'number') {
        throw new TypeError(`Invalid input, numeric value is required for [${i}, ${j}]`);
      }
    }
  }
}

export function matrixEqual(x: Matrix, y: Matrix, tol = 0): boolean {
  if (x.shape[0] !== y.shape[0] || x.shape[1] !== y.shape[1]) {
    return false;
  }
  if (max2d(abs2d(sub2d(x, y)), -1).values() > tol) {
    return false;
  }
  return true;
}

export function vectorEqual(x: Vector, y: Vector, tol = 0): boolean {
  if (x.length !== y.length) {
    return false;
  }
  if (max1d(abs1d(sub1d(x, y))).values() > tol) {
    return false;
  }
  return true;
}

export function scalarEqual(x: Scalar, y: Scalar, tol = 0): boolean {
  if (Math.abs(x.values() - y.values()) > tol) {
    return false;
  }
  return true;
}
