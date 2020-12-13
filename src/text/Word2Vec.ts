import * as tf from '@tensorflow/tfjs';
import { createData, createUniqueWord, objectLength, skipGram, UniqueWord } from './word2vec-utils';

export default class Word2Vec {
  public weight: number[][] | undefined = undefined;
  public uniqueWords: UniqueWord;
  private data: number[][];
  private label: number[];
  private nwords: number;
  private size: number;

  /**
   *
   * @param text Array of strings
   * @param window window size for skip-gram
   * @param stopwords
   * @param size embedding dimension
   */
  constructor(text: string[], window: number, stopwords: string[] | undefined, size = 50) {
    const [ wordList, allText ] = skipGram(window, text, stopwords);
    this.uniqueWords = createUniqueWord(allText as string[]);
    this.nwords = objectLength(this.uniqueWords);
    const [ data, label ] = createData(wordList as string[][], this.uniqueWords, this.nwords);
    this.data = data as number[][];
    this.label = label as number[];
    this.size = size;
  }

  /**
   * Train word2vec model
   * @param epochs
   * @param verbose to print out the training logs or not
   * @return Promise
   */
  public async train(epochs = 100, verbose = 0): Promise<void> {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: this.size, inputShape: [ this.nwords ] }));
    model.add(tf.layers.dense({ units: this.nwords, activation: 'softmax' }));
    model.compile({ optimizer: 'sgd', loss: 'sparseCategoricalCrossentropy', metrics: [ 'accuracy' ] });

    const dataTensor = tf.tensor(this.data);
    const labelTensor = tf.tensor(this.label);

    await model.fit(dataTensor, labelTensor, {
      epochs: epochs,
      callbacks: {
        onEpochEnd: (epoch, log) => {
          if (verbose){
            console.log(`Epoch ${epoch}: loss = ${log.loss}`);
          }
        }
      }
    });
    this.weight = model.getWeights()[0].arraySync() as number[][];
  }

  /**
   *Calculate the similarity between two words.
   * @param wordA
   * @param wordB
   * @return number
   */
  public similarity(wordA: string, wordB: string): number {

    if ( !(wordA in this.uniqueWords && wordB in this.uniqueWords)) {
      throw new Error("word does not exist");
    }

    const indexA = this.uniqueWords[wordA];
    const indexB = this.uniqueWords[wordB];
    const weightA = tf.tensor(this.weight[indexA]);
    const weightB = tf.tensor(this.weight[indexB]);
    const sim = weightA.sub(weightB).pow(2).mean().sqrt();
    return sim.arraySync() as number;

  }

  /**
   * Obtain word weight vector.
   * @param word
   * @return number[]
   */
  public getWeight(word: string): number[] {
    if (!(word in this.uniqueWords)) {
      throw new Error('word does not exist');
    }

    const index = this.uniqueWords[word];
    return this.weight[index];
  }

  /**
   * Calculate most similar words to a given word in a descending order
   * @param word
   * @return Array<Array<string | number>>
   */
  public mostSimilar(word: string): Array<Array<string | number>> {
    if (!(word in this.uniqueWords)) {
      throw new Error('word does not exist');
    }

    const weight = this.weight.slice();
    const index = this.uniqueWords[word];
    const wordWeight = tf.tensor(this.weight[index]);
    weight.splice(index - 1, 1);

    const rslt = tf.tensor(weight).sub(wordWeight).pow(2).mean(1).sqrt();
    const simArray = rslt.arraySync() as number[];

    let similarities = Object.keys(this.uniqueWords);
    similarities.splice(index - 1, 1); // remove word from uniquewords

    let sim: Array<Array<string | number>> = similarities.map((value, i) => {
      return [ value, simArray[i] ];
    });
    //sort array in descending order
    sim.sort((a, b) => {
      let sa = a[1] as number;
      let sb = b[1] as number;
      return sb - sa;
    });

    return sim;
  }

}
