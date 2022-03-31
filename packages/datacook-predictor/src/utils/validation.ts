export function checkJsArray2D(arr: number[][]): void {
  if (!arr || !arr.length) {
    throw new TypeError('Invalid input, two dimensional array is expected');
  }
  if (!(Array.isArray(arr[0]))) {
    throw new TypeError('Invalid input, two dimensional array is expected');
  }
  const m = arr[0].length;
  for (let i = 0; i < arr.length; i++) {
    if (!(Array.isArray(arr[i])) || arr[i].length !== m) {
      throw new TypeError('Invalid input, array length should be the same on each row');
    }
    for (let j = 0; j < arr[i].length; j++) {
      if (typeof arr[i][j] !== 'number') {
        throw new TypeError(`Invalid input, numeric value is required for [${i}, ${j}]`);
      }
    }
  }
}
