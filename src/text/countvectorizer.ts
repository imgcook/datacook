import Counter from './counter';

interface StringIntegerObject {
  [property: string] : number
}
export default class CountVectorizer {
  public wordOrder: StringIntegerObject = {};
  public uniqueLength: number;
  /**
   * Convert text to vector base on their number of count
   * and the order to which they occur alphabetically.
   * @param textArray
   */
  constructor(textArray: string[]) {

    const tokenArray: string[] = [];
    textArray.forEach((value) => {
      value.split(' ').forEach((text) => {
        if (text != '') {
          tokenArray.push(text.toLocaleLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, ''));
        }
      });
    });

    const uniqueWord = Array.from(new Set(tokenArray));
    const sortedUniqueWord: string[] = this.sort(uniqueWord);

    // store array element as object element
    // to enable quick search of element
    // when transforming input arrays.
    sortedUniqueWord.forEach((text, index) => {
      if (text !== '') {
        this.wordOrder[text] = index;
      }
    });
    this.uniqueLength = sortedUniqueWord.length;

  }

  private sort(textArray: string[]): string[] {
    return textArray.sort((a: string, b: string) => {
      return a.charCodeAt(0) - b.charCodeAt(0);
    });
  }
  /**
   * Transform input text to vector.
   * @param textArray
   * @returns number[][] array of number (vectors)
   */
  public transform(textArray: string[]): number[][] {

    const counterVectorizer: number[][] = textArray.map((value) => {
      const innerArray: number[] = Array.from(new Float64Array(this.uniqueLength));
      const cleanTextArray: string[] = [];
      value.split(' ').forEach((text) => {
        if (text !== '') {
          const cleanText = text.toLocaleLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, '');
          cleanTextArray.push(cleanText);
        }
      });

      const wordCounter = new Counter(cleanTextArray).wordCount;
      cleanTextArray.forEach((text) => {
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

