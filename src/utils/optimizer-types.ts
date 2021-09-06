/**
 * Types definination for tfjs optimizers
 */
import { SGDOptimizer, MomentumOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer, AdamaxOptimizer, RMSPropOptimizer, train } from '@tensorflow/tfjs-core';
export type OptimizerType = 'sgd' | 'momentum' | 'adagrad' | 'adadelta' | 'adam' | 'adamax' | 'rmsprop';
export type Optimizer = SGDOptimizer | MomentumOptimizer | AdagradOptimizer | AdadeltaOptimizer | AdamaxOptimizer | AdamOptimizer | RMSPropOptimizer;
export type SGDProps = {
  learningRate: number
};
export type MomentumProps = {
  learningRate: number,
  momentum: number,
  useNesterov?: boolean
};
export type AdagradProps = {
  learningRate: number,
  initialAccumulatorValue: number
}
export type AdadeltaProps = {
  learningRate?: number,
  rho?: number,
  epsilon?: number
}
export type AdamProps = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number
}
export type AdamaxProps = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number,
  decay?: number
}
export type RMSPropProps = {
  learningRate: number,
  decay?: number,
  momentum?: number,
  epsilon?: number,
  centered?: boolean
}
export type OptimizerProps = SGDProps | MomentumProps | AdagradProps | AdadeltaProps | AdamProps | AdamaxProps | RMSPropProps;

/**
 * Create optimizer depending on given type and props
 * @param optimizerTypes optimizer types
 * @param optimizerProps optimizer properties
 */
export const getOptimizer = (optimizerTypes: OptimizerType, optimizerProps: OptimizerProps): Optimizer => {
  switch (optimizerTypes) {
  case 'sgd': {
    const props = (optimizerProps as SGDProps);
    if (props) {
      return train.sgd(props.learningRate);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'momentum':{
    const props = (optimizerProps as MomentumProps);
    if (props) {
      const { learningRate, momentum, useNesterov } = props;
      return train.momentum(learningRate, momentum, useNesterov);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adagrad': {
    const props = (optimizerProps as AdagradProps);
    if (props) {
      const { learningRate, initialAccumulatorValue } = props;
      return train.adagrad(learningRate, initialAccumulatorValue);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adadelta': {
    const props = (optimizerProps as AdadeltaProps);
    if (props) {
      const { learningRate, rho, epsilon } = props;
      return train.adadelta(learningRate, rho, epsilon);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adam': {
    const props = (optimizerProps as AdamProps);
    if (props) {
      const { learningRate, beta1, beta2, epsilon } = props;
      return train.adam(learningRate, beta1, beta2, epsilon);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adamax': {
    const props = (optimizerProps as AdamaxProps);
    if (props) {
      const { learningRate, beta1, beta2, epsilon, decay } = props;
      return train.adamax(learningRate, beta1, beta2, epsilon, decay);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'rmsprop': {
    const props = (optimizerProps as RMSPropProps);
    if (props) {
      const { learningRate, decay, momentum, epsilon, centered } = props;
      return train.rmsprop(learningRate, decay, momentum, epsilon, centered);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  }
};
