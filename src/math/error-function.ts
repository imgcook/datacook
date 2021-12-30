/**
 * Compute error function by approximation
 * @param x input number
 * @return erf(x)
 */
export const errorFunction = (x: number): number => {
  const p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736,
    a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
  const t = 1 / (1 + (p * Math.abs(x)));
  const erf = 1 - (a1 * t
    + a2 * Math.pow(t, 2)
    + a3 * Math.pow(t, 3)
    + a4 * Math.pow(t, 4)
    + a5 * Math.pow(t, 5)) * Math.pow(Math.E, -Math.pow(x, 2));
  // using the fact of odd function, erf x = -erf(-x)
  return (x >= 0) ? erf : -erf;
};
