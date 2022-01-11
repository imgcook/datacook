
require('@tensorflow/tfjs-backend-cpu');
const { range, reshape, ones } = require('@tensorflow/tfjs-core');

describe('Generic Split test in browser', () => {
  it('should split data into train & test', async () => {
<<<<<<< HEAD
    const X = reshape(range(0, 10), [5, 2]);
    const y = range(0, 5);
    const [ X_train, X_test, y_train, y_test ] = datacook.Generic.split([X, y]);

    const actX_test = reshape(range(6, 10), [2, 2]);
    const actX_train = reshape(range(0, 6), [3, 2]);

    const acty_train = reshape(range(0, 3), [3]);
    const acty_test = reshape(range(3, 5), [2]);
=======
    const X = reshape(range(0, 10), [ 5, 2 ]);
    const y = range(0, 5);
    const [ X_train, X_test, y_train, y_test ] = datacook.Generic.split([ X, y ]);

    const actX_test = reshape(range(6, 10), [ 2, 2 ]);
    const actX_train = reshape(range(0, 6), [ 3, 2 ]);

    const acty_train = reshape(range(0, 3), [ 3 ]);
    const acty_test = reshape(range(3, 5), [ 2 ]);
>>>>>>> 00617e14762fdb30bc36b981821576c5e7911148

    expect(actX_test.dataSync()).to.eql(X_test.dataSync());
    expect(actX_train.dataSync()).to.eql(X_train.dataSync());

    expect(acty_train.dataSync()).to.eql(y_train.dataSync());
    expect(acty_test.dataSync()).to.eql(y_test.dataSync());
  });

<<<<<<< HEAD
  it('should throws if tensors length equals zero ', () => { 
    expect(() => datacook.Generic.split([])).to.throw('inputs should not have length of zero');
  }); 

  it('should throws if inputs have different first dimension', () => { 
    const t1 = ones([10, 28, 28, 3]);
    const t2 = ones([8]);
    expect(() => datacook.Generic.split([t1, t2]))
      .to.throw('inputs should have the same length');
  }); 
=======
  it('should throws if tensors length equals zero ', () => {
    expect(() => datacook.Generic.split([])).to.throw('inputs should not have length of zero');
  });

  it('should throws if inputs have different first dimension', () => {
    const t1 = ones([ 10, 28, 28, 3 ]);
    const t2 = ones([ 8 ]);
    expect(() => datacook.Generic.split([ t1, t2 ]))
      .to.throw('inputs should have the same length');
  });
>>>>>>> 00617e14762fdb30bc36b981821576c5e7911148
});
