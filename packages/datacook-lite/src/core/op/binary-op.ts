import { Matrix, Vector } from "../classes";
import {
  add2d as add2dCpu,
  sub2d as sub2dCpu,
  mul2d as mul2dCpu,
  div2d as div2dCpu,
  add1d as add1dCpu,
  sub1d as sub1dCpu,
  mul1d as mul1dCpu,
  div1d as div1dCpu
} from '../../backend-cpu/op/binary-op';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";

export const add2d = (x: Matrix, y: Matrix | Vector | number, by = 0): Matrix => {
  if (IS_CPU_BACKEND) {
    return add2dCpu(x, y, by);
  } else {
    throw new TypeError(getMethodErrorStr('add2d'));
  }
};

export const sub2d = (x: Matrix, y: Matrix | Vector | number, by = 0): Matrix => {
  if (IS_CPU_BACKEND) {
    return sub2dCpu(x, y, by);
  } else {
    throw new TypeError(getMethodErrorStr('sub2d'));
  }
};

export const mul2d = (x: Matrix, y: Matrix | Vector | number, by = 0): Matrix => {
  if (IS_CPU_BACKEND) {
    return mul2dCpu(x, y, by);
  } else {
    throw new TypeError(getMethodErrorStr('mul2d'));
  }
};

export const div2d = (x: Matrix, y: Matrix | Vector | number, by = 0): Matrix => {
  if (IS_CPU_BACKEND) {
    return div2dCpu(x, y, by);
  } else {
    throw new TypeError(getMethodErrorStr('div2d'));
  }
};

export const add1d = (x: Vector, y: Vector | number): Vector => {
  if (IS_CPU_BACKEND) {
    return add1dCpu(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('add1d'));
  }
};

export const sub1d = (x: Vector, y: Vector | number): Vector => {
  if (IS_CPU_BACKEND) {
    return sub1dCpu(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('sub1d'));
  }
};

export const mul1d = (x: Vector, y: Vector | number): Vector => {
  if (IS_CPU_BACKEND) {
    return mul1dCpu(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('mul1d'));
  }
};

export const div1d = (x: Vector, y: Vector | number): Vector => {
  if (IS_CPU_BACKEND) {
    return div1dCpu(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('div1d'));
  }
};
