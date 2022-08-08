/**
 * Reference: https://github.com/substack.gamma.js
 */
const g = 7.0;
const p = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7
];

const gLn = 607 / 128.0;
const pLn = [
  0.99999999999999709182,
  57.156235665862923517,
  -59.597960355475491248,
  14.136097974741747174,
  -0.49191381609762019978,
  0.33994649984811888699e-4,
  0.46523628927048575665e-4,
  -0.98374475304879564677e-4,
  0.15808870322491248884e-3,
  -0.21026444172410488319e-3,
  0.21743961811521264320e-3,
  -0.16431810653676389022e-3,
  0.84418223983852743293e-4,
  -0.26190838401581408670e-4,
  0.36899182659531622704e-5
];

/**
 * Approximation for lnGamma for large number
 * @param z input number
 * @returns approximation of lnGamma for large number
 */
export const lnGammaForLargeNumber = (z: number): number => {
  if (z < 0) return Number('0/0');
  let x = pLn[0];
  for (let i = pLn.length - 1; i > 0; --i) {
    x += pLn[i] / (z + i);
  }
  const t = z + gLn + 0.5;
  return .5 * Math.log(2 * Math.PI) + (z + .5) * Math.log(t) - t + Math.log(x) - Math.log(z);
};

/**
 * Calculate gamma function using Lanczcos approximation.
 * https://en.wikipedia.org/wiki/Lanczos_approximation
 * @param z input variable
 * @returns gamma(z)
 */
export const gamma = (z: number): number => {
  if (z < 0.5) {
    return Math.PI / (Math.sin(Math.PI * z) * gamma(1.0 - z));
  } else {
    if (z > 100) return Math.exp(lnGammaForLargeNumber(z));
    else {
      z -= 1;
      let x = p[0];
      for (let i = 1; i < g + 2; i++) {
        x += p[i] / (z + i);
      }
      const t = z + g + 0.5;
      return Math.sqrt(2 * Math.PI)
        * Math.pow(t, z + 0.5)
        * Math.exp(-t)
        * x;
    }
  }
};

export const lnGamma = (x: number): number => {
  return Math.log(gamma(x));
};
