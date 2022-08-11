import { square2d, sub2d, sum2d } from '../../src/backend-cpu/op';
import { matrix } from '../../src/core/classes';
import { SGDOptimizer } from '../../src/core/optimizer/sgd-optimizer';

describe('SGDOptimizer', () => {
    it('optimize', () => {
      const a = matrix([ [ 0, 0 ], [ 0, 0 ] ]);
      const b = matrix([ [ 3, 4 ], [ 5, 6 ] ]);
      const distLoss = () => {
        return sum2d(square2d(sub2d(a, b)), -1);
      }
      const optimizer = new SGDOptimizer([ a ], { learningRate: 0.1 });
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
