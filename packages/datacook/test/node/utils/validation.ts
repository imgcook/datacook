import { assert } from 'chai';
import 'mocha';
import { checkArray } from '../../../src/utils/validation';

describe('Validation', function () {
  describe('ArrayChecker', () => {

    it('convert array to int32 tensor', () => {
      const array = [ 1, 2, 3 ];
      const arrayTensor = checkArray(array, 'int32');
      arrayTensor.print();
      assert.equal(arrayTensor.dtype, 'int32');
    });

    it('convert two dimension array to int32 tensor', () => {
      const array = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ];
      const arrayTensor = checkArray(array, 'int32');
      arrayTensor.print();
      assert.equal(arrayTensor.dtype, 'int32');
    });

  });
});
