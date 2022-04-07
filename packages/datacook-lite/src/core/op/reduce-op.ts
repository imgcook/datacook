import { Matrix, Vector } from "../classes";
import {
  argMax2d as argMax2dCPU,
  argMin2d as argMin2dCPU,
  max2d as max2dCPU,
  min2d as min2dCPU,
  sum2d as sum2dCPU,
  mean2d as mean2dCPU
} from '../../backend-cpu/op/reduce-op';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";

export const argMax2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return argMax2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('argMax2d'));
  }
};

export const argMin2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return argMin2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('argMin2d'));
  }
};

export const max2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return max2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('max2d'));
  }
};

export const min2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return min2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('min2d'));
  }
};

export const sum2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return sum2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('sum2d'));
  }
};

export const mean2d = (x: Matrix, by = 0): number | Vector => {
  if (IS_CPU_BACKEND) {
    return mean2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('mean2d'));
  }
};
