export declare class Beta {
  /**
   * New a Beta Distribution Generator
   * @param a
   * @param b
   */
  constructor(a?: number, b?:number);

  /**
   * Sample a value from Beta Distribution
   */
  generate(): number;

  /**
   * Set a new set of a & b param
   * @param a Alpha
   * @param b Beta
   */
  setParam(a: number, b: number): void;
}
