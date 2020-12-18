import Counter from './counter';

interface StringIntegerObject {
  [property: string] : number
}
export default class CountVectorizer {
  public wordOrder: StringIntegerObject = {};
  public uniqueLength: number;

  constructor(textArray: string[]) {

    const tokenArray: string[] = [];
    textArray.forEach((value) => {
      value.split(" ").forEach((text) => {
        if (text != '') {
          tokenArray.push(text.toLocaleLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, ''));
        }
      });
    });

    const uniqueWord = Array.from(new Set(tokenArray));
    const sortedUniqueWord: string[] = this.sort(uniqueWord);

    // console.log(sortedUniqueWord);
    // store array element as object element
    // to enable quick search of element
    // when transforming input arrays.
    sortedUniqueWord.forEach((text, index) => {
      if (text != ''){
        this.wordOrder[text] = index;
      }
    });
    this.uniqueLength = sortedUniqueWord.length;

  }

  private sort(textArray: string[]): string[] {
    const collator = new Intl.Collator(undefined, { numeric:true, sensitivity: 'base' });

    return textArray.sort(collator.compare);
  }

  public transform(textArray: string[]): number[][] {

    const counterVectorizer: number[][] = textArray.map((value) => {
      const innerArray: number[] = Array.from(new Float64Array(this.uniqueLength));
      const ar: string[] = [];
      value.split(" ").forEach((text) => {
        if (text != '') {
          const cleanText = text.toLocaleLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '');
          ar.push(cleanText);
        }

      });

      const wordCounter = new Counter(ar).wordCount;
      ar.forEach((text) => {
        if (text in this.wordOrder) {
          const wordIndex = this.wordOrder[text];
          const value = wordCounter[text];
          innerArray[wordIndex] = value;
        }
      });
      return innerArray;
    });
    return counterVectorizer;

  }
}

