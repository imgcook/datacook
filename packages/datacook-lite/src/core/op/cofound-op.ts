import { Matrix } from "../classes";
import { getMethodErrorStr } from "./utils";
import { IS_CPU_BACKEND } from "../../env";
import { ByAxis } from "../../backend-cpu/op/basic-impl";

import {
  softmax2d as softmax2dCPU
} from '../../backend-cpu/op/cofound-op';

export const softmax2d = (x: Matrix, by: ByAxis = ByAxis.ByRow): Matrix => {
  if (IS_CPU_BACKEND) {
    return softmax2dCPU(x, by);
  } else {
    throw new TypeError(getMethodErrorStr('sub2d'));
  }
};
