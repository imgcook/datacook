export interface Denpendency {
  gradFunc: (data: Tensor) => Tensor;
  target: Tensor;
}


type RecursiveArray = number | number[] | number[][] | number[][][] | number[][][][] | number[][][][][];

export abstract class Tensor {
  public data: RecursiveArray;
  public shape: number | number[];
  public grad: Tensor;
  public dependency: Array<Denpendency>;
  abstract backward(grad?: Tensor): void;
  abstract values(): RecursiveArray;
}

export abstract class Scalar extends Tensor {
  public data: number;
  public grad: Scalar;
  public constructor(data: number) {
    super();
    this.data = data;
    this.dependency = [];
  }
  abstract values(): number;
  abstract backward(grad?: Scalar): void;
}

export abstract class Matrix extends Tensor {
  public data: number[][];
  public shape: number[];
  public grad: Matrix;
  public dependency: Array<Denpendency>;
  public constructor(data: number[][]) {
    super();
    this.shape = [ data.length, data[0].length ];
    this.dependency = [];
  }
  abstract values(): number[][];
  abstract getColumn(i: number): Vector;
  abstract getRow(i: number): Vector;
  abstract get(i: number, j: number): number;
  abstract setColumn(i: number, x: Vector | number[]): void;
  abstract set(i: number, j: number, x: number): void;
  abstract backward(grad?: Matrix): void;
}


export abstract class Vector extends Tensor{
  public data: number[];
  public length: number;
  public grad: Vector;
  public dependency: Array<Denpendency>;
  constructor(data: number[]) {
    super();
    this.data = data;
    this.length = this.data.length;
    this.dependency = [];
  }
  abstract values(): number[];
  abstract get(i: number): number;
  abstract set(i: number, val: number): void;
  abstract backward(grad?: Vector): void;
}

