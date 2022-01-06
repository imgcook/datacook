
import { sort } from '../../../../src/model/tree/utils';
import { assert } from 'chai';
const x = [ 5, 3, 6, 1, 5, 2, 6, 43, 234, 65 ];
const xSorted = [ 1, 2, 3, 5, 5, 6, 6, 43, 65, 234];
const indices = Array.from(new Array(x.length).keys());
const indicesSorted = [ 3, 5, 1, 4, 0, 2, 6, 7, 9, 8 ];
describe('IntroSort', () => {
  it('sort simple', () => {
    sort(x, indices, 0, x.length);
    let eq = true;
    let eqInd = true;
    for (let i = 0; i < x.length; i++) {
      if (x[i] !== xSorted[i]) {
        eq = false;
      }
      if (indices[i] !== indicesSorted[i]) {
        eqInd = false;
      }
    }
    assert.isTrue(eq && eqInd);
  });
});
  