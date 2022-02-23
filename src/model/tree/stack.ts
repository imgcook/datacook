export interface StackRecord {
  start: number;
  end: number;
  depth: number;
  parent: number;
  isLeft: boolean;
  impurity: number;
  nConstantFeatures: number;
}

export class Stack {
  public capacity: number;
  public top: number;
  public stack: StackRecord[];
  constructor(capacity: number) {
    this.capacity = capacity;
    this.top = 0;
    this.stack = [];
  }
  public isEmpty(): boolean {
    return this.stack.length <= 0;
  }
  /**
   * Push a new element onto a stack
   */
  public push(stackRecord: StackRecord): void {
    const {
      start,
      end,
      depth,
      parent,
      isLeft,
      impurity,
      nConstantFeatures
    } = stackRecord;
    if (this.top >= this.capacity) {
      this.capacity *= 2;
    }
    this.stack.push({
      start,
      end,
      depth,
      parent,
      isLeft,
      impurity,
      nConstantFeatures
    });
    this.top = this.top + 1;
  }
  public pop(): StackRecord {
    return this.stack.pop();
  }
}
