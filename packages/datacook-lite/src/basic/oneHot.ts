export const oneHot = (x: number[], size: number): number[][] => {
  return x.map((d: number) => {
    const dOneHot = new Array(size).fill(0);
    dOneHot[d] = 1;
    return dOneHot;
  });
};
