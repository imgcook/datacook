import { assert } from 'chai';
import CountVectorizer from "../../../src/text/countvectorizer";

describe('CountVectorizer', function(){
  it('should convert words to their proper countvectorizer', function(){
    const vectorizer = new CountVectorizer(["The quick brown fox jumped over the lazy big fat dog."], ['big', 'fat']);
    const toVector = vectorizer.transform(["the brown brown","the the brown hat car"]);
    const vocab = {
                    brown: 0,
                    dog: 1,
                    fox: 2,
                    jumped: 3,
                    lazy: 4,
                    over: 5,
                    quick: 6,
                    the: 7
                  };
    const expectedVector = [
                            [2, 0, 0, 0,0, 0, 0, 1],
                            [1, 0, 0, 0,0, 0, 0, 2]
                           ];

    assert.deepEqual(vectorizer.wordOrder, vocab);
    assert.deepEqual(toVector, expectedVector);
  });
  it('should convert words array to their proper countvectorizer', function(){
    const vectorizer = new CountVectorizer([['The'],['quick'], ['brown'], ['fox'], ['jumped'], ['over'], ['the'], ['lazy'], ['dog']], ['the']);
    const toVector = vectorizer.transform(["the brown brown","the the brown"]);
    const vocab = {
                    brown: 0,
                    dog: 1,
                    fox: 2,
                    jumped: 3,
                    lazy: 4,
                    over: 5,
                    quick: 6
                  };
    const expectedVector = [
                            [2, 0, 0, 0,0, 0, 0],
                            [1, 0, 0, 0,0, 0, 0]
                           ];
    console.log(vectorizer.wordOrder);
    assert.deepEqual(vectorizer.wordOrder, vocab);
    assert.deepEqual(toVector, expectedVector);
  });
});
