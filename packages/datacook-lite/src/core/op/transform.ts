import { Matrix } from "../classes";
import {
  transpose2d as transpose2dCPU
} from '../../backend-cpu/op/transform';
import { IS_CPU_BACKEND } from "../../env";
import { getMethodErrorStr } from "./utils";

export const transpose2d = (x: Matrix): Matrix => {
  if (IS_CPU_BACKEND) {
    return transpose2dCPU(x);
  } else {
    throw new TypeError(getMethodErrorStr('transpose2d'));
  }
};
