import { assert, expect } from "chai"
import { createData, createUniqueWord, objectLength, skipGram } from '../../../src/text/word2vec-utils';
import { text_list, stopwords, text_lower, text} from './global';
import '@tensorflow/tfjs-backend-cpu';

describe("Word2Vec Utils", () => {
  
  it("should generate skip gram using window =5", () => {
    
    const [ wordList, allText ] = skipGram(5, text_list, stopwords);
    const expectedWordList = [
                              [ 'king', 'man' ],
                              [ 'king', 'rules' ],
                              [ 'king', 'over' ],
                              [ 'king', 'nation' ],
                              [ 'king', 'he' ],
                              [ 'man', 'rules' ],
                              [ 'man', 'king' ],
                              [ 'man', 'over' ],
                              [ 'man', 'nation' ],
                              [ 'man', 'he' ]
                            ]
    const expectedWordListLength = 190;
    const expectedAllText = [
                              'king',      'man',      'rules',
                              'over',      'nation',   'he',
                              'always',    'have',     'woman',
                              'beside',    'him',      'called',
                              'queen',     'she',      'helps',
                              'king',      'controls', 'affars',
                              'nation',    'perhaps',  'she',
                              'acclaimed', 'position', 'king',
                              'king',      'her',      'husband',
                              'deceased'
                            ]

    assert.deepEqual(wordList.slice(0,10), expectedWordList);
    assert.deepEqual(wordList.length, expectedWordListLength);
    assert.deepEqual(allText, expectedAllText);
  });

  it('should obtain number of element in an object', () => {
    const obj = {
      "King": 2,
      "Queen": 4,
      "Lady": 6,
      "woman": 8,
      "gh": 9
    }
    const length = 5;
    assert.deepEqual(objectLength(obj), length);
  });

  it("should generate Unique word dictionary", () => {
    const textList = [
                      'king',      'man',      'rules',
                      'over',      'nation',   'he',
                      'always',    'have',     'woman',
                      'beside',    'him',      'called',
                      'queen',     'she',      'helps',
                      'king',      'controls', 'affars',
                      'nation',    'perhaps',  'she',
                      'acclaimed', 'position', 'king',
                      'king',      'her',      'husband',
                      'deceased'
                    ]
    const uniqueWords = {
                          king: 1,
                          man: 2,
                          rules: 3,
                          over: 4,
                          nation: 5,
                          he: 6,
                          always: 7,
                          have: 8,
                          woman: 9,
                          beside: 10,
                          him: 11,
                          called: 12,
                          queen: 13,
                          she: 14,
                          helps: 15,
                          controls: 16,
                          affars: 17,
                          perhaps: 18,
                          acclaimed: 19,
                          position: 20,
                          her: 21,
                          husband: 22,
                          deceased: 23
                        }
    assert.deepEqual(createUniqueWord(textList), uniqueWords);
  });

  it("should generate dataset for word2vec", () => {
    const [ wordList, allText ] = skipGram(5, text_list, stopwords);
    const uniqueWords = createUniqueWord(allText as string[]);
    const nwords = objectLength(uniqueWords);
    const [ data, label ] = createData(wordList as string[][], uniqueWords, nwords);

    assert.equal(data.length, 190);
    assert.equal(label.length, 190);
  });
});