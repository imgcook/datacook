import { assert, expect } from 'chai'
import Word2Vec from '../../src/text/word2vec';
import { text_list, stopwords} from './global';

describe("Word2Vec", () => {
  it('should contain necessary uniquewords', () => {
    const word2vec = new Word2Vec(text_list, 5, stopwords);
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
    assert.deepEqual(word2vec.uniqueWords, uniqueWords);
  })
  it('should contain the right weight dimension', async () => {
    const word2vec = new Word2Vec(text_list, 5, stopwords);

    await word2vec.train();
    const expectedShape = [23,50];
    const weightShape = [word2vec.weight.length, word2vec.weight[0].length];

    assert.deepEqual(weightShape, expectedShape)
  })
  it('should calculate the similarity between words', async () => {
    const word2vec = new Word2Vec(text_list, 5, stopwords);

    await word2vec.train();
    
    expect(word2vec.similarity('king', 'man')).to.lessThan(1.0);
  })
  it('should obtain weight vector for word', async ()=>{
    const word2vec = new Word2Vec(text_list, 5, stopwords);

    await word2vec.train();
    expect(word2vec.getWeight('king').length).to.eq(50)
  })
  it('should obtain the most similar words', async () => {
    const word2vec = new Word2Vec(text_list, 5, stopwords);

    await word2vec.train();
    expect(word2vec.mostSimilar('king').length).to.eq(22);
  })
});
