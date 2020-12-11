import { assert } from 'chai';
import Counter from "../../src/text/counter";

describe("Counter", ()=>{

  it('should generate the correct word count dictionary for array input', ()=>{
    const counter = new Counter(["good", 'good', "bad", "bad", "neural", "neural","neural"]);
    const expectedWordCount = { 'good': 2, 'bad': 2, 'neural': 3 }
    const expectedElement = [ [ 'neural', 3 ], [ 'good', 2 ], [ 'bad', 2 ] ]
    
    assert.deepEqual(counter.wordCount,expectedWordCount);
    assert.deepEqual(counter.element, expectedElement);
  });
  it('should generate the correct word count dictionary for dictionary input', ()=>{
    const counter = new Counter({ 'good': 2, 'bad': 2, 'neural': 3 });
    const expectedWordCount = { 'good': 2, 'bad': 2, 'neural': 3 }
    const expectedElement = [ [ 'neural', 3 ], [ 'good', 2 ], [ 'bad', 2 ] ]
    
    assert.deepEqual(counter.wordCount,expectedWordCount);
    assert.deepEqual(counter.element, expectedElement);
  });
  it('should generate the correct word count dictionary for dictionary input', ()=>{
    const counter = new Counter("awesome and amazing work");
    const expectedWordCount = {
                            a: 4,
                            w: 2,
                            e: 2,
                            s: 1,
                            o: 2,
                            m: 2,
                            ' ': 3,
                            n: 2,
                            d: 1,
                            z: 1,
                            i: 1,
                            g: 1,
                            r: 1,
                            k: 1
                          }
    const expectedElement = [
                              [ 'a', 4 ], [ ' ', 3 ],
                              [ 'w', 2 ], [ 'e', 2 ],
                              [ 'o', 2 ], [ 'm', 2 ],
                              [ 'n', 2 ], [ 's', 1 ],
                              [ 'd', 1 ], [ 'z', 1 ],
                              [ 'i', 1 ], [ 'g', 1 ],
                              [ 'r', 1 ], [ 'k', 1 ]
                            ]
    assert.deepEqual(counter.wordCount,expectedWordCount);
    assert.deepEqual(counter.element, expectedElement);
  });
  it('should update word count', ()=>{
    const counter = new Counter({ 'good': 2, 'bad': 2, 'neural': 3 });
    counter.update({"good": 4,"bad":3, "food":2});

    const expectedWordCount = { 'good': 6, 'bad': 5, 'neural': 3, 'food':2};
    const expectedElement = [ [ 'good', 6 ], [ 'bad', 5 ], [ 'neural', 3 ], [ 'food', 2 ] ]
    
    assert.deepEqual(counter.wordCount,expectedWordCount);
    assert.deepEqual(counter.element, expectedElement);
  });
  it('should fetch the top 2 common word', ()=>{
    const counter = new Counter({ 'good': 1, 'bad': 2, 'neural': 3 });
    const expectedElement = [ [ 'neural', 3 ], [ 'bad', 2 ] ]
    assert.deepEqual(counter.mostCommon(2), expectedElement);
  });
});