import { square2d, sub2d, sum2d } from '../../src/backend-cpu/op';
import { matrix } from '../../src/core/classes';
import { mul2d } from '../../src/core/op';
import { AdaDeltaOptimizer } from '../../src/core/optimizer/adadelta-optimizer';

describe('AdaDeltaOptimizer', () => {
    it('optimize', () => {
      const a = matrix([ [ 0, 0 ], [ 0, 0 ] ]);
      const b = matrix([ [ 3, 4 ], [ 5, 6 ] ]);
      const distLoss = () => {
        return sum2d(square2d(sub2d(mul2d(a, 2), b)), -1);
      }
      const optimizer = new AdaDeltaOptimizer([ a ], { learningRate: 1, rho: 0.5 });
      for (let i = 0; i < 100; i++) {
        optimizer.zeroGrad();
        const d = distLoss();
        d.backward();
        optimizer.step();
        if (i % 10 == 0) {
            console.log('step ', i);
            console.log('loss', d.values());
            console.log('a = ', a.values());
        }
      }
    });
});
