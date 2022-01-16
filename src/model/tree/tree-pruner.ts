abstract class PruneController {
  abstract stopPruning(effectiveAlpha: number): boolean;
  // abstract saveMetrics(effectiveAlpha: number, subtreeImpurities: number): void;
  abstract afterPruning(inSubtree: boolean[]): void;
}

export class AlphaPruner extends PruneController {
  public alpha: number;
  public capacity: number;

  constructor(alpha: number) {
    super();
    this.alpha = alpha;
    this.capacity = 0;
  }

  public stopPruning(effectiveAlpha: number): boolean {
    return this.alpha < effectiveAlpha;
  }

  public afterPruning(inSubtree: boolean[]): void {
    for (let i = 0; i < inSubtree.length; i++) {
      if (inSubtree[i]) {
        this.capacity += 1;
      }
    }
  }
}
