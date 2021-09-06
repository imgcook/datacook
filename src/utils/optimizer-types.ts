/**
 * Types definination for tfjs optimizers
 */
import { SGDOptimizer, MomentumOptimizer, AdagradOptimizer, AdadeltaOptimizer, AdamOptimizer, AdamaxOptimizer, RMSPropOptimizer } from '@tensorflow/tfjs-core'
export type OptimizerTypes = 'sgd' | 'momentum' | 'autograd' | 'adadelta' | 'adam' | 'adamax' | 'rmsprop'
export type Optimizer = SGDOptimizer | MomentumOptimizer | AdagradOptimizer | AdadeltaOptimizer | AdamaxOptimizer | AdamOptimizer | RMSPropOptimizer
export interface SGDPropsTypes{
  learningRate: number
};
export type MomentumPropsTypes = {
  learningRate: number,
  momentum: number,
  useNesterov?: boolean
};
export type AdagradPropsTypes = {
  learningRate: number,
  initialAccumulatorValue: number
}
export type AdadeltaPropsTypes = {
  learningRate?: number,
  rho?: number,
  epsilon?: number
}
export type AdamPropsTypes = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number
}
export type AdamaxPropsTypes = {
  learningRate?: number,
  beta1?: number,
  beta2?: number,
  epsilon?: number,
  decay?: number
}
export type RMSPropPropsTypes = {
  learningRate: number,
  decay?: number,
  momentum?: number,
  epsilon?: number,
  centered?: boolean
}
export type OptimizerPropsTypes = SGDPropsTypes | AdagradPropsTypes | AdadeltaPropsTypes | AdamPropsTypes | AdamaxPropsTypes | RMSPropPropsTypes;
