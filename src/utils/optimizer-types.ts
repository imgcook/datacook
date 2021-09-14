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
};
export type AdadeltaProps = {
  learningRate?: number,
  rho?: number,
  epsilon?: number
};
export type AdamProps = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number
};
export type AdamaxProps = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number,
  decay?: number
};
export type RMSPropProps = {
  learningRate: number,
  decay?: number,
  momentum?: number,
  epsilon?: number,
  centered?: boolean
};

export type OptimizerProps = SGDProps | MomentumProps | AdagradProps | AdadeltaProps | AdamProps | AdamaxProps | RMSPropProps;

function isSGDProps(arg: any): arg is SGDProps {
  return arg?.learningRate > 0;
}

function isMomentumProps(arg: any): arg is MomentumProps {
  return arg?.learningRate > 0 && arg?.momentum > 0;
}

function isAdagradProps(arg: any): arg is AdadeltaProps {
  return arg?.learningRate > 0 && arg?.initialAccumulatorValue > 0;
}

function isAdadeltaProps(arg: any): arg is AdadeltaProps {
  return true;
}

function isAdamProps(arg: any): arg is AdamProps {
  return true;
}

function isAdaMaxProps(arg: any): arg is AdamProps {
  return true;
}

function isRMSPropProps(arg: any): arg is RMSPropProps {
  return arg?.learningRate > 0;
}
const optimizerTypeAssertMap: Record<string, any> = {
  sgd: isSGDProps,
  momentum: isMomentumProps,
  adagrad: isAdagradProps,
  adadelta: isAdadeltaProps,
  adam: isAdamProps,
  adamax: isAdaMaxProps,
  rmsprop: isRMSPropProps
};
/**
 * Create optimizer depending on given type and props
 * @param optimizerType optimizer type
 * @param optimizerProps optimizer properties
 */
export const getOptimizer = (optimizerType: OptimizerType, optimizerProps: OptimizerProps): Optimizer => {
  const typeCheck = optimizerTypeAssertMap[optimizerType];
  if (typeCheck && !typeCheck(optimizerProps)) {
    throw new TypeError('Illegal properties to init optimizer');
  }
  switch (optimizerType) {
  case 'sgd':
    return train.sgd((optimizerProps as SGDProps).learningRate);
  case 'momentum': {
    const { learningRate, momentum, useNesterov } = optimizerProps as MomentumProps;
    return train.momentum(learningRate, momentum, useNesterov);
  }
  case 'adagrad': {
    const { learningRate, initialAccumulatorValue } = optimizerProps as AdagradProps;
    return train.adagrad(learningRate, initialAccumulatorValue);
  }
  case 'adadelta': {
    const { learningRate, rho, epsilon } = optimizerProps as AdadeltaProps;
    return train.adadelta(learningRate, rho, epsilon);
  }
  case 'adam': {
    const { learningRate, beta1, beta2, epsilon } = optimizerProps as AdamProps;
    return train.adam(learningRate, beta1, beta2, epsilon);
  }
  case 'adamax': {
    const { learningRate, beta1, beta2, epsilon, decay } = optimizerProps as AdamaxProps;
    return train.adamax(learningRate, beta1, beta2, epsilon, decay);
  }
  case 'rmsprop': {
    const { learningRate, decay, momentum, epsilon, centered } = optimizerProps as RMSPropProps;
    return train.rmsprop(learningRate, decay, momentum, epsilon, centered);
  }
  default:
    throw new TypeError('Illegal optimizer type: ' + optimizerType);
  }
};
