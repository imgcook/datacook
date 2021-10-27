import { Tensor, abs, sub, norm } from "@tensorflow/tfjs-core";
import { shapeEqual } from "../linalg";

export const euclideanDistance = (x: Tensor, y: Tensor): number => {
  if (!shapeEqual(x, y)) {
    throw new TypeError('Shape of input tensor should be equal');
  }
  return norm(sub(x, y)).dataSync()[0];
};

export const manhattanDistance = (x: Tensor, y: Tensor): number {
  
}
