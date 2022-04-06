import { Matrix, Vector } from "../../classes";

export const add2dcustomize = (x: Matrix, y: Matrix | Vector | number, by = 0): Matrix => { 
  if (process.env.BACKEND === 'cpu') {
      const ops = require('../cpu/binary-op');
      // console.log(ops);
      return ops.add2d(x, y, by);
    } else {
      const ops = require('../webgpu/binary-op');
      // console.log(ops);
      console.log('gpu');
      return ops.add2d(x, y, by);
      // return require('../cpu/').sub2d(x, y, by);
    }
};