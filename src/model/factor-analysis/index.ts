/**
 * Factor Analysis.
 *
 * Factor analysis is a decomposition algorithm which similar to probabilistic PCA.
 * In factor analysis, visible vector variable v is related to the vector hidden
 * variable h by a linear mapping, with independent additive Gaussian noise on each
 * visible variable.
 *
 * The implementation of factor analysis uses an EM-based method to find the best
 * solution for factor loading.
 */
 import { svd } from "../../linalg";

 export type FactorAnalysisParams = {
   nComponent: number,
   tol?: number,
   maxIterTimes?: number
 }
 
 export class FactorAnalysis {
   public nComponents: number;
   public tol: number;
 
   constructor(params: FactorAnalysisParams) {
 
   }
 
   public fit(xData: ) {
 
   }
 }
 
 