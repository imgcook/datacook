import { gamma } from './gamma';

/**
 * Copmution of beta function
 * Beta(x, y) = Gamma(x) * Gamma(y) / Gamma(x + y)
 * @returns Beta(x, y)
 */
export const beta = (x: number, y: number): number => {
  return gamma(x) * gamma(y) / gamma(x + y);
};

const TINY = 1e-8;
const STOP = 1e-30;

/**
 * Regularized Incomplete Beta Function.
 * Reference: https://github.com/codeplea/incbeta
 * @returns B(x; a, b)
 */
export const incbeta = (a: number, b: number, x: number): number => {
  if (x < 0.0 || x > 1.0) return 1.0 / 0.0;

  /* The continued fraction converges nicely for x < (a+1)/(a+b+2) */
  if (x > (a + 1.0) / (a + b + 2.0)) {
    return (1.0 - incbeta(b, a, 1.0 - x)); /* Use the fact that beta is symmetrical. */
  }

  /* Find the first part before the continued fraction. */
  const front = Math.exp(Math.log(x) * a + Math.log(1.0 - x) * b) / beta(a, b) / a;
  /* Use Lentz's algorithm to evaluate the continued fraction. */
  let f = 1.0, c = 1.0, d = 0.0;

  for (let i = 0; i <= 200; ++i) {
    const m = Math.floor(i / 2);
    let numerator: number;
    if (i === 0) {
      /* First numerator is 1.0. */
      numerator = 1.0;
    } else if (i % 2 === 0) {
      /* Even term. */
      numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
    } else {
      /* Odd term. */
      numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
    }

    /* Do an iteration of Lentz's algorithm. */
    d = 1.0 + numerator * d;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1.0 / d;

    c = 1.0 + numerator / c;
    if (Math.abs(c) < TINY) c = TINY;

    const cd = c * d;
    f *= cd;

    /* Check for stop. */
    if (Math.abs(1.0 - cd) < STOP) {
      return front * (f - 1.0);
    }
  }
  /* Needed more loops, did not converge. */
  return 1.0 / 0.0;
};
