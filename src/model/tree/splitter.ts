import { Criterion } from './criterion';
export class Splitter {
  public criterion: Criterion;
  public nSamples: number;
  public nFeatures: number;
  public featureValues: number[];
  public maxFeatures: number;
  public minSampleLeaf: number;
  public minWeightLeaf: number;

}
