export const unique = <T>(x: Array<T>): T[] => {
  const uniques: T[] = [];
  x.forEach((xi: T) => {
    if(uniques.indexOf(xi) == -1) {
      uniques.push(xi);
    }
  });
  return uniques;
}