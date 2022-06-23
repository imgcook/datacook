import { Matrix, Vector } from "../classes";
import {
  sqrt2d as sqrt2dCPU,
  square2d as square2dCPU,
  exp2d as exp2dCPU,
  pow2d as pow2dCPU,
  neg2d as neg2dCPU,
  abs2d as abs2dCPU,
  sigmoid2d as sigmoid2dCPU,
  sqrt1d as sqrt1dCPU
} from '../../backend-cpu/op/single-op';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";

export const square2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return square2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('square2d'));
  }
};

export const sqrt2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return sqrt2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('sqrt2d'));
  }
};


export const exp2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return exp2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('exp2d'));
  }
};

export const pow2d = (x: Matrix, y: number): Matrix => {
  if (IS_CPU_BACKEND) {
    return pow2dCPU(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('pow2d'));
  }
};


export const neg2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return neg2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('neg2d'));
  }
};

export const abs2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return abs2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('abs2d'));
  }
};

export const sigmoid2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return sigmoid2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('sigmoid2d'));
  }
};

export const sqrt1d = (x: Vector): Vector => {
  if (IS_CPU_BACKEND) {
    return sqrt1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('sqrt1d'));
  }
};
