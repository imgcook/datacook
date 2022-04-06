import { Matrix } from "../../src/classes";
import { add2dcustomize } from "../../src/op/interface";

describe('OP Test', () => {
  const a = new Matrix([[1, 2],[3, 4]]);
  const c = add2d(a, 2);
  console.log(c);
});