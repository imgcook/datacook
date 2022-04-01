import { Matrix, Vector } from "../../classes";

export const squeeze = (x: Matrix): Vector => {
  let out: number[] = [];
  const nX = x.shape[0];
  for (let i = 0; i < nX; i++) {
    out = [ ...out, ...x.getRow(i).data ];
  }
  return new Vector(out);
};

// export const transpose2d = (x: Matrix): Matrix => {

// }
