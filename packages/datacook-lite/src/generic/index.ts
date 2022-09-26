export function shuffle(inputs: Array<any>): void {
  for (let i = inputs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [ inputs[i], inputs[j] ] = [ inputs[j], inputs[i] ];
  }
}
