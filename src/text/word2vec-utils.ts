export function clean(text: string, stopwords: string[] | undefined): Array<string> {
  let textArray = text.split(' ');
  //filter out empty string
  textArray = textArray.filter((value) => {
    return value != '';
  });

  //filter out stopwords.
  if (stopwords) {
    textArray = textArray.filter((value) => {
      return !stopwords.includes(value);
    });
  }
  //remove punctuation
  textArray = textArray.map((value) => {
    return value.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
  });

  return textArray;
}

export function skipGram(
  window = 5, textArray: Array<string>,
  stopwords: string[] | undefined): Array<Array<string | string[]>> {

  let wordList = [];
  let allText = [];

  for (const index in textArray) {
    let text = textArray[index];
    let textClean = clean(text, stopwords);

    allText.push(...textClean); //store clean text
    for (let i = 0; i < textClean.length; i++) {
      let word = textClean[i];
      for (let j = 0; j < window; j++) {
        //skip-gram forward
        if ((i + 1 + j) < textClean.length) {
          wordList.push([ word, textClean[i + 1 + j] ]);
        }
        //skip-gram backward
        if ((i - j - 1) >= 0) {
          wordList.push([ word, textClean[i - j - 1] ]);
        }
      }
    }
  }
  return [ wordList, allText ];
}

export interface UniqueWord {
  [word: string] : number;
}

export function createUniqueWord(textArray: string[]): UniqueWord {
  const wordSet = new Set(textArray);
  const wordList = Array.from(wordSet);
  let uniqueWords: UniqueWord = {};

  for (let i = 0; i < wordList.length; i++) {
    const word = wordList[i];
    uniqueWords[word] = i + 1;
  }
  return uniqueWords;
}

export function objectLength(dict: UniqueWord): number {
  return Object.keys(dict).length;
}

export function createData(
  wordList: string[][], uniqueWords: UniqueWord,
  nWords: number): Array<Array<number | number[]>> {

  let data = [];
  let label = [];

  for (let i = 0; i < wordList.length; i++) {
    const [ x, y ] = wordList[i];
    const wordIndex = uniqueWords[x];
    const contextIndex = uniqueWords[y];

    let xRow = new Float64Array(nWords);
    xRow[wordIndex] = 1;
    data.push(Array.from(xRow));
    label.push(contextIndex);
  }

  return [ data, label ];
}
