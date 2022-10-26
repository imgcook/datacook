export interface MetricParams {
  p?: number;
}
export abstract class DistanceMetric {
  public p: number;
  public name: string;
  abstract dist(x1: number[], x2: number[]): number;
  abstract rDist(x1: number[], x2: number[]): number;
  abstract distToRDist(dist: number): number;
  abstract rDistToDist(rDist: number): number;
}

export class MinkowskiDistance extends DistanceMetric {
  constructor(p: number) {
    super();
    this.p = p;
    this.name = 'minkowski';
  }
  public rDist(x1: number[], x2: number[]): number {
    return x1.map((d, i) => Math.pow(Math.abs(d - x2[i]), this.p)).reduce((a, b) => a + b);
  }
  public dist(x1: number[], x2: number[]): number {
    return Math.pow(this.rDist(x1, x2), 1 / this.p);
  }
  public distToRDist(dist: number): number {
    return Math.pow(dist, this.p);
  }
  public rDistToDist(rDist: number): number {
    return Math.pow(rDist, 1 / this.p);
  }
}

export class EuclideanDistance extends DistanceMetric {
  constructor() {
    super();
    this.p = 2;
    this.name = 'euclidean';
  }
  public rDist(x1: number[], x2: number[]): number {
    return x1.map((d, i) => Math.pow(d - x2[i], 2)).reduce((a, b) => a + b);
  }
  public dist(x1: number[], x2: number[]): number {
    return Math.pow(this.rDist(x1, x2), 1 / 2);
  }
  public distToRDist(dist: number): number {
    return Math.pow(dist, 2);
  }
  public rDistToDist(rDist: number): number {
    return Math.pow(rDist, 1 / 2);
  }
}

export class ManhattanDistance extends DistanceMetric {
  constructor() {
    super();
    this.p = 2;
    this.name = 'manahttan';
  }
  public rDist(x1: number[], x2: number[]): number {
    return x1.map((d, i) => Math.abs(d - x2[i])).reduce((a, b) => a + b);
  }
  public dist(x1: number[], x2: number[]): number {
    return this.rDist(x1, x2);
  }
  public distToRDist(dist: number): number {
    return dist;
  }
  public rDistToDist(rDist: number): number {
    return rDist;
  }
}

export type MetricName = 'euclidean' | 'l2' | 'l1' | 'manhattan' | 'minkowski'

export class MetricFactory {
  public static getMetric(name: MetricName, params?: MetricParams): DistanceMetric {
    switch (name) {
    case ('euclidean'):
    case ('l2'): {
      return new EuclideanDistance();
    }
    case ('manhattan'):
    case ('l1'): {
      return new ManhattanDistance();
    }
    case ('minkowski'): {
      const { p } = params;
      return new MinkowskiDistance(p);
    }
    default:
      return new EuclideanDistance();
    }
  }
}
