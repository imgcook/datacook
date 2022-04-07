import { Matrix } from "../classes";
import {
  matMul2d as matmul2dCPU
} from '../../backend-cpu/op/matmul';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";

export const matMul2d = (x: Matrix, y: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return matmul2dCPU(x, y);
  } else {
    throw new TypeError(getMethodErrorStr('matmul2d'));
  }
};
