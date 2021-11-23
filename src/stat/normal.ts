import { errorFunction } from "../math/error-function";
/**
 * Cumulative density function of random varaible
 * @param x input number
 * @param loc mean
 * @param scale standard deviation
 * @returns cdf for x in mean `loc` and standard deviation `scale`
 */
export const cdf = (x: number, loc = 0, scale = 1): number => {
  return 0.5 * (1 + errorFunction((x - loc) / (scale * Math.SQRT2)));
};
