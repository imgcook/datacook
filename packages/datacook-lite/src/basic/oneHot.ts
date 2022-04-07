export const oneHot = (x: number[], size: number) => {
  return x.map((d: number) => {
    let dOneHot = new Array(size).fill(0);
    dOneHot[d] = 1;
    return dOneHot;
  })
}