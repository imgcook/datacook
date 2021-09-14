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
  if (arg?.learningRate > 0) {
    return true;
  }
  return false;
}

function isMomentumProps(arg: any): arg is MomentumProps {
  if (arg?.learningRate > 0 && arg?.momentum > 0) {
    return true;
  }
  return false;
}

function isAdagradProps(arg: any): arg is AdadeltaProps {
  if (arg?.learningRate > 0 && arg?.initialAccumulatorValue > 0) {
    return true;
  }
  return false;
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
  if (arg?.learningRate > 0) {
    return true;
  }
  return false;
}

/**
 * Create optimizer depending on given type and props
 * @param optimizerType optimizer type
 * @param optimizerProps optimizer properties
 */
export const getOptimizer = (optimizerType: OptimizerType, optimizerProps: OptimizerProps): Optimizer => {
  switch (optimizerType) {
  case 'sgd': {
    const props = isSGDProps(optimizerProps) ? (optimizerProps as SGDProps) : null;
    if (props) {
      return train.sgd(props.learningRate);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'momentum':{
    const props = isMomentumProps(optimizerProps) ? (optimizerProps as MomentumProps) : null;
    if (props) {
      const { learningRate, momentum, useNesterov } = props;
      return train.momentum(learningRate, momentum, useNesterov);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adagrad': {
    const props = isAdagradProps(optimizerProps) ? (optimizerProps as AdagradProps) : null;
    if (props) {
      const { learningRate, initialAccumulatorValue } = props;
      return train.adagrad(learningRate, initialAccumulatorValue);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adadelta': {
    const props = isAdadeltaProps(optimizerProps) ? (optimizerProps as AdadeltaProps) : null;
    if (props) {
      const { learningRate, rho, epsilon } = props;
      return train.adadelta(learningRate, rho, epsilon);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adam': {
    const props = isAdamProps(optimizerProps) ? (optimizerProps as AdamProps) : null;
    if (props) {
      const { learningRate, beta1, beta2, epsilon } = props;
      return train.adam(learningRate, beta1, beta2, epsilon);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'adamax': {
    const props = isAdaMaxProps(optimizerProps) ? (optimizerProps as AdamaxProps) : null;
    if (props) {
      const { learningRate, beta1, beta2, epsilon, decay } = props;
      return train.adamax(learningRate, beta1, beta2, epsilon, decay);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  case 'rmsprop': {
    const props = isRMSPropProps(optimizerProps) ? (optimizerProps as RMSPropProps) : null;
    if (props) {
      const { learningRate, decay, momentum, epsilon, centered } = props;
      return train.rmsprop(learningRate, decay, momentum, epsilon, centered);
    } else {
      throw new TypeError('Illegal properties to init optimizer');
    }
  }
  default: {
    throw new TypeError('Illegal optimizer type: ' + optimizerType);
  }
  }
};
