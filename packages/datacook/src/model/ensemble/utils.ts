export const randomSampleMask = (nTotalSamples: number, nTotalInBag: number): number[] => {
  const sampleMask = new Array(nTotalSamples).fill(0);
  let nBagged = 0;
  for (let i = 0; i < nTotalSamples; i++) {
    const rand = Math.random();
    if (rand * (nTotalSamples - i) < nTotalInBag - nBagged) {
      sampleMask[i] = 1;
      nBagged += 1;
    }
  }
  return sampleMask;
};
