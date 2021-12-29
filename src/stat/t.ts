
import { incbeta } from '../math/beta';
import * as Normal from './normal';
const cdfNormal = Normal.cdf;
/**
 * Cumulative density function of student distribution
 * @param x input number
 * @param df degree of freedom
 * @returns cdf of x in degree of freedom `df`
 */
export const cdf = (x: number, df: number): number => {
  const xSquare = x * x;
  const f = (x + Math.sqrt(xSquare + df)) / (2.0 * Math.sqrt(xSquare + df));
  if (df <= 100) {
    return incbeta(df / 2.0, df / 2.0, f);
  } else {
    /**
     * use the fact that when df close to infinity,
     * t distribution approximates to standard normal distribution
<<<<<<< HEAD
     *  */ 
=======
     **/
>>>>>>> ee5144fdea4675d2f98e809bb71db970622eb7f0
    return cdfNormal(x);
  }
};
