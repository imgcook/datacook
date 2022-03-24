/**
 * Check if two numbers are equal in a given tolerence.
 * @param a input number a
 * @param b input number b
 * @param tol tolerence, default is 0
 * @returns If a and b are equal in tolerence `tol`.
 */
export const numEqual = (a: number, b: number, tol = 0): boolean => {
  if (a !== Infinity)
    return Math.abs(a - b) <= tol;
  return a === b;
};

/**
 * Determine whether a number is integer or not
 * @param a input number
 * @returns true for integer, false for not
 */
export const isInteger = (a: number): boolean => {
  return typeof a === 'number' && a % 1 === 0;
};
