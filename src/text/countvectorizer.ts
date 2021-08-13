import Counter from './counter';

interface StringIntegerObject {
  [property: string] : number
}
type TextInput = string | string[];

export default class CountVectorizer {
  public wordOrder: StringIntegerObject = {};
  public uniqueLength: number;
  public stopWords: string[] = [];
  /**
   * Convert text to vector base on their number of count
   * and the order to which they occur alphabetically.
   * @param textArray could be one or two dimension string array. In the case of two dimension array, each row of data should be the words list after words cutting.
   * @param stopWords stop words
   */
  constructor(textArray: TextInput[], stopWords: string[] = []) {

    let tokenArray: string[] = [];

    textArray.forEach((value: TextInput) => {
      let wordElements: string[] = [];
      if (Array.isArray(value)) {
        wordElements = value;
      } else {
        wordElements = value.split(' ');
      }
      wordElements.forEach((text) => {
        if (text !== '') {
          tokenArray.push(text.toLocaleLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, ''));
        }
      });
    });

    const uniqueWord = Array.from(new Set(tokenArray)).filter((d) => stopWords.indexOf(d) == -1);
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
  public transform(textArray: TextInput[]): number[][] {

    const counterVectorizer: number[][] = textArray.map((value: TextInput) => {
      const innerArray: number[] = Array.from(new Float64Array(this.uniqueLength));
      const cleanTextArray: string[] = [];
      let wordElements: string[] = [];
      if (Array.isArray(value)) {
        wordElements = value;
      } else {
        wordElements = value.split(' ');
      }
      wordElements.forEach((text) => {
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

