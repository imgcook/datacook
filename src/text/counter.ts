// eslint-disable-next-line @typescript-eslint/no-unused-vars
interface CounterObject {
  [property: string] : number
}

type CounterType = CounterObject | string[] | string;


export default class Counter {
  public wordCount: CounterObject = {};
  public element: Array<Array<string | number>> ;

  /**
   * Initializer Counter class
   * create WordCOunt dictionary and array of element
   * with containing word and their count in order of
   * increasing decreaing value
   * e.g
   * const counter = new Counter(["a","boy","girl"])
   * counter.wordCount // {'a':1, "boy":1, "girl":1}
   *
   * @param textInput
   */
  constructor (textInput: CounterType) {
    if (Array.isArray(textInput)) {
      this.wordCountFromArray(textInput);

    } else if (typeof textInput === "string") {
      const toArray = textInput.split('');
      this.wordCountFromArray(toArray);

    } else {
      this.wordCount = textInput;
    }
    //create order list in Ascending order
    this.createOrderedElement();

  }

  /**
   * Update a counter dictionary
   * e.g
   * const counter = new Counter(["a","boy","girl"])
   * counter.update({'a':2,'boy':4,'eat':2})
   * counter.wordCount // {'a': 3, 'boy':5, 'girl':1,'eat':2}
   *
   * @param updateInput
   */
  public update(updateInput: CounterType): void {
    const updateCounter = new Counter(updateInput);

    for (const key in updateCounter.wordCount) {
      if ( key in this.wordCount) {
        this.wordCount[key] += updateCounter.wordCount[key];
      } else {
        this.wordCount[key] = updateCounter.wordCount[key];
      }
    }
    this.createOrderedElement();
  }

  /**
   * Generate wordCount dictionary from array
   * @param textArray string[]
   */
  private wordCountFromArray(textArray: string[]) : void {
    for (let index = 0; index < textArray.length; index++) {
      const elem = textArray[index];
      if (elem in this.wordCount) {
        this.wordCount[elem] += 1;
      } else {
        this.wordCount[elem] = 1;
      }
    }
  }

  /**
   * create an array of words and their count
   * arranged in descending order
   * e.g
   * [['a',30],['z',20],['x',12] ....]
   */
  private createOrderedElement() : void {

    this.element = Object.keys(this.wordCount)
      .sort((a, b) => {
        return this.wordCount[b] - this.wordCount[a];
      })
      .map((x) => [ x, this.wordCount[x] ]);
  }

  /**
   * Fetch the most common words base on
   * the number specified.
   * @param count number
   * @return Array<Array<string | number>>
   */
  public mostCommon(count: number): Array<Array<string | number>> {
    return this.element.slice(0, count);
  }
}


