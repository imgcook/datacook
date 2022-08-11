import { Matrix, Vector } from "../classes";
import {
  argMax2d as argMax2dCPU,
  argMin2d as argMin2dCPU,
  max2d as max2dCPU,
  min2d as min2dCPU,
  sum2d as sum2dCPU,
  mean2d as mean2dCPU,
  argMax1d as argMax1dCPU,
  argMin1d as argMin1dCPU,
  max1d as max1dCPU,
  min1d as min1dCPU,
  sum1d as sum1dCPU,
  mean1d as mean1dCPU

} from '../../backend-cpu/op';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";
import { Scalar } from "../../backend-cpu/classes";
import { ByAxis } from "../../backend-cpu/op/basic-impl";

export const argMax2d = (x: Matrix, by = 0): Scalar | Vector | Matrix => {
  if (IS_CPU_BACKEND) {
    return argMax2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('argMax2d'));
  }
};

export const argMin2d = (x: Matrix, by = 0): Scalar | Vector | Matrix => {
  if (IS_CPU_BACKEND) {
    return argMin2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('argMin2d'));
  }
};

export const max2d = (x: Matrix, by = 0): Scalar | Vector => {
  if (IS_CPU_BACKEND) {
    return max2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('max2d'));
  }
};

export const min2d = (x: Matrix, by = 0): Scalar | Vector => {
  if (IS_CPU_BACKEND) {
    return min2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('min2d'));
  }
};

export const sum2d = (x: Matrix, by: ByAxis = 0): Scalar | Vector => {
  if (IS_CPU_BACKEND) {
    return sum2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('sum2d'));
  }
};

export const mean2d = (x: Matrix, by: ByAxis = 0): Scalar | Vector => {
  if (IS_CPU_BACKEND) {
    return mean2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('mean2d'));
  }
};

export const argMax1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return argMax1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('argMax1d'));
  }
};

export const argMin1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return argMin1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('argMin1d'));
  }
};

export const min1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return min1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('min1d'));
  }
};

export const max1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return max1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('max1d'));
  }
};

export const sum1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return sum1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('sum1d'));
  }
};

export const mean1d = (x: Vector): Scalar => {
  if (IS_CPU_BACKEND) {
    return mean1dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('mean1d'));
  }
};
