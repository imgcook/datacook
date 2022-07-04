import { BestSplitter } from '../../../../src/model/tree/splitter';
import { EntropyCriterion } from '../../../../src/model/tree/criterion';
import { DepthFirstTreeBuilder } from '../../../../src/model/tree/tree-builder';
import { Tree } from '../../../../src/model/tree/tree';
import { accuracyScore, getRSquare } from '../../../../src/metrics';
import { DecisionTreeClassifier, DecisionTreeRegressor } from '../../../../src/model/tree';
import { assert } from 'chai';
const irisData = [
  [ 5.1, 3.5, 1.4, 0.2 ],
  [ 4.9, 3., 1.4, 0.2 ],
  [ 4.7, 3.2, 1.3, 0.2 ],
  [ 4.6, 3.1, 1.5, 0.2 ],
  [ 5., 3.6, 1.4, 0.2 ],
  [ 5.4, 3.9, 1.7, 0.4 ],
  [ 4.6, 3.4, 1.4, 0.3 ],
  [ 5., 3.4, 1.5, 0.2 ],
  [ 4.4, 2.9, 1.4, 0.2 ],
  [ 4.9, 3.1, 1.5, 0.1 ],
  [ 5.4, 3.7, 1.5, 0.2 ],
  [ 4.8, 3.4, 1.6, 0.2 ],
  [ 4.8, 3., 1.4, 0.1 ],
  [ 4.3, 3., 1.1, 0.1 ],
  [ 5.8, 4., 1.2, 0.2 ],
  [ 5.7, 4.4, 1.5, 0.4 ],
  [ 5.4, 3.9, 1.3, 0.4 ],
  [ 5.1, 3.5, 1.4, 0.3 ],
  [ 5.7, 3.8, 1.7, 0.3 ],
  [ 5.1, 3.8, 1.5, 0.3 ],
  [ 5.4, 3.4, 1.7, 0.2 ],
  [ 5.1, 3.7, 1.5, 0.4 ],
  [ 4.6, 3.6, 1., 0.2 ],
  [ 5.1, 3.3, 1.7, 0.5 ],
  [ 4.8, 3.4, 1.9, 0.2 ],
  [ 5., 3., 1.6, 0.2 ],
  [ 5., 3.4, 1.6, 0.4 ],
  [ 5.2, 3.5, 1.5, 0.2 ],
  [ 5.2, 3.4, 1.4, 0.2 ],
  [ 4.7, 3.2, 1.6, 0.2 ],
  [ 4.8, 3.1, 1.6, 0.2 ],
  [ 5.4, 3.4, 1.5, 0.4 ],
  [ 5.2, 4.1, 1.5, 0.1 ],
  [ 5.5, 4.2, 1.4, 0.2 ],
  [ 4.9, 3.1, 1.5, 0.2 ],
  [ 5., 3.2, 1.2, 0.2 ],
  [ 5.5, 3.5, 1.3, 0.2 ],
  [ 4.9, 3.6, 1.4, 0.1 ],
  [ 4.4, 3., 1.3, 0.2 ],
  [ 5.1, 3.4, 1.5, 0.2 ],
  [ 5., 3.5, 1.3, 0.3 ],
  [ 4.5, 2.3, 1.3, 0.3 ],
  [ 4.4, 3.2, 1.3, 0.2 ],
  [ 5., 3.5, 1.6, 0.6 ],
  [ 5.1, 3.8, 1.9, 0.4 ],
  [ 4.8, 3., 1.4, 0.3 ],
  [ 5.1, 3.8, 1.6, 0.2 ],
  [ 4.6, 3.2, 1.4, 0.2 ],
  [ 5.3, 3.7, 1.5, 0.2 ],
  [ 5., 3.3, 1.4, 0.2 ],
  [ 7., 3.2, 4.7, 1.4 ],
  [ 6.4, 3.2, 4.5, 1.5 ],
  [ 6.9, 3.1, 4.9, 1.5 ],
  [ 5.5, 2.3, 4., 1.3 ],
  [ 6.5, 2.8, 4.6, 1.5 ],
  [ 5.7, 2.8, 4.5, 1.3 ],
  [ 6.3, 3.3, 4.7, 1.6 ],
  [ 4.9, 2.4, 3.3, 1. ],
  [ 6.6, 2.9, 4.6, 1.3 ],
  [ 5.2, 2.7, 3.9, 1.4 ],
  [ 5., 2., 3.5, 1. ],
  [ 5.9, 3., 4.2, 1.5 ],
  [ 6., 2.2, 4., 1. ],
  [ 6.1, 2.9, 4.7, 1.4 ],
  [ 5.6, 2.9, 3.6, 1.3 ],
  [ 6.7, 3.1, 4.4, 1.4 ],
  [ 5.6, 3., 4.5, 1.5 ],
  [ 5.8, 2.7, 4.1, 1. ],
  [ 6.2, 2.2, 4.5, 1.5 ],
  [ 5.6, 2.5, 3.9, 1.1 ],
  [ 5.9, 3.2, 4.8, 1.8 ],
  [ 6.1, 2.8, 4., 1.3 ],
  [ 6.3, 2.5, 4.9, 1.5 ],
  [ 6.1, 2.8, 4.7, 1.2 ],
  [ 6.4, 2.9, 4.3, 1.3 ],
  [ 6.6, 3., 4.4, 1.4 ],
  [ 6.8, 2.8, 4.8, 1.4 ],
  [ 6.7, 3., 5., 1.7 ],
  [ 6., 2.9, 4.5, 1.5 ],
  [ 5.7, 2.6, 3.5, 1. ],
  [ 5.5, 2.4, 3.8, 1.1 ],
  [ 5.5, 2.4, 3.7, 1. ],
  [ 5.8, 2.7, 3.9, 1.2 ],
  [ 6., 2.7, 5.1, 1.6 ],
  [ 5.4, 3., 4.5, 1.5 ],
  [ 6., 3.4, 4.5, 1.6 ],
  [ 6.7, 3.1, 4.7, 1.5 ],
  [ 6.3, 2.3, 4.4, 1.3 ],
  [ 5.6, 3., 4.1, 1.3 ],
  [ 5.5, 2.5, 4., 1.3 ],
  [ 5.5, 2.6, 4.4, 1.2 ],
  [ 6.1, 3., 4.6, 1.4 ],
  [ 5.8, 2.6, 4., 1.2 ],
  [ 5., 2.3, 3.3, 1. ],
  [ 5.6, 2.7, 4.2, 1.3 ],
  [ 5.7, 3., 4.2, 1.2 ],
  [ 5.7, 2.9, 4.2, 1.3 ],
  [ 6.2, 2.9, 4.3, 1.3 ],
  [ 5.1, 2.5, 3., 1.1 ],
  [ 5.7, 2.8, 4.1, 1.3 ],
  [ 6.3, 3.3, 6., 2.5 ],
  [ 5.8, 2.7, 5.1, 1.9 ],
  [ 7.1, 3., 5.9, 2.1 ],
  [ 6.3, 2.9, 5.6, 1.8 ],
  [ 6.5, 3., 5.8, 2.2 ],
  [ 7.6, 3., 6.6, 2.1 ],
  [ 4.9, 2.5, 4.5, 1.7 ],
  [ 7.3, 2.9, 6.3, 1.8 ],
  [ 6.7, 2.5, 5.8, 1.8 ],
  [ 7.2, 3.6, 6.1, 2.5 ],
  [ 6.5, 3.2, 5.1, 2. ],
  [ 6.4, 2.7, 5.3, 1.9 ],
  [ 6.8, 3., 5.5, 2.1 ],
  [ 5.7, 2.5, 5., 2. ],
  [ 5.8, 2.8, 5.1, 2.4 ],
  [ 6.4, 3.2, 5.3, 2.3 ],
  [ 6.5, 3., 5.5, 1.8 ],
  [ 7.7, 3.8, 6.7, 2.2 ],
  [ 7.7, 2.6, 6.9, 2.3 ],
  [ 6., 2.2, 5., 1.5 ],
  [ 6.9, 3.2, 5.7, 2.3 ],
  [ 5.6, 2.8, 4.9, 2. ],
  [ 7.7, 2.8, 6.7, 2. ],
  [ 6.3, 2.7, 4.9, 1.8 ],
  [ 6.7, 3.3, 5.7, 2.1 ],
  [ 7.2, 3.2, 6., 1.8 ],
  [ 6.2, 2.8, 4.8, 1.8 ],
  [ 6.1, 3., 4.9, 1.8 ],
  [ 6.4, 2.8, 5.6, 2.1 ],
  [ 7.2, 3., 5.8, 1.6 ],
  [ 7.4, 2.8, 6.1, 1.9 ],
  [ 7.9, 3.8, 6.4, 2. ],
  [ 6.4, 2.8, 5.6, 2.2 ],
  [ 6.3, 2.8, 5.1, 1.5 ],
  [ 6.1, 2.6, 5.6, 1.4 ],
  [ 7.7, 3., 6.1, 2.3 ],
  [ 6.3, 3.4, 5.6, 2.4 ],
  [ 6.4, 3.1, 5.5, 1.8 ],
  [ 6., 3., 4.8, 1.8 ],
  [ 6.9, 3.1, 5.4, 2.1 ],
  [ 6.7, 3.1, 5.6, 2.4 ],
  [ 6.9, 3.1, 5.1, 2.3 ],
  [ 5.8, 2.7, 5.1, 1.9 ],
  [ 6.8, 3.2, 5.9, 2.3 ],
  [ 6.7, 3.3, 5.7, 2.5 ],
  [ 6.7, 3., 5.2, 2.3 ],
  [ 6.3, 2.5, 5., 1.9 ],
  [ 6.5, 3., 5.2, 2. ],
  [ 6.2, 3.4, 5.4, 2.3 ],
  [ 5.9, 3., 5.1, 1.8 ]
];

const labels = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ];
const label_ids = labels.map((d) => d - 1);


describe('DepthFirstTreeBuilder', () => {
  it('build tree', () => {
    const criterion = new EntropyCriterion();
    const splitter = new BestSplitter(criterion, 4, 3, 3);
    const treeBuilder = new DepthFirstTreeBuilder(splitter, 3, 3, 3, 4, 0);
    const tree = new Tree(4, 3);
    treeBuilder.build(tree, irisData, label_ids);
    for (let i = 0; i < tree.nodeCount;i++){
      console.log(JSON.stringify(tree.nodes[i]));
    }
  });
});

describe('DecisionTreeClassifier', () => {
  it('fit iris', async () => {
    const dt = new DecisionTreeClassifier();
    await dt.fit(irisData, labels);
    const predY = await dt.predict(irisData);
    const acc = accuracyScore(labels, predY);
    console.log('accuracy score: ', acc);
    assert.isTrue(acc > 0.95);
  });

  it('predict probability', async () => {
    const dt = new DecisionTreeClassifier();
    await dt.fit(irisData, labels);
    const predProba = await dt.predictProb(irisData);
    // const acc = accuracyScore(labels, predY);
    // console.log('accuracy score: ', acc);
    // assert.isTrue(acc > 0.95);
  });

  it('build pruned tree', async () => {
    const dt = new DecisionTreeClassifier({ ccpAlpha: 0.01 });
    await dt.fit(irisData, labels);
    const predY = await dt.predict(irisData);
    const acc = accuracyScore(labels, predY);
    console.log('accuracy score: ', acc);
    assert.isTrue(dt.tree.nodeCount < 17);
  });

  it('save and load model (classifier)', async () => {
    const dt = new DecisionTreeClassifier({ ccpAlpha: 0.01 });
    await dt.fit(irisData, labels);
    
    const modelJson = await dt.toJson();
    const dt2 = new DecisionTreeClassifier();
    dt2.fromJson(modelJson);
    const predY = await dt2.predict(irisData);
    const acc = accuracyScore(labels, predY);
    console.log('accuracy score: ', acc);
    assert.isTrue(dt.tree.nodeCount < 17);
  });
});

describe('DecisionTreeRegressor', () => {
  it('fit iris', async () => {
    const dt = new DecisionTreeRegressor();
    const features = irisData.map((d) => [ d[0], d[1], d[2] ]);
    const target = irisData.map((d) => d[3]);
    await dt.fit(features, target);
    const predY = await dt.predict(features);
  });
  it('save and load model (regressor)', async () => {
    const dt = new DecisionTreeRegressor();
    const features = irisData.map((d) => [ d[0], d[1], d[2] ]);
    const target = irisData.map((d) => d[3]);
    await dt.fit(features, target);
    const dt2 = new DecisionTreeRegressor();
    const modelJson = await dt.toJson();
    await dt2.fromJson(modelJson);
    const predY = await dt2.predict(features);
    const r2 = getRSquare(target, predY as number[]);
    console.log('r2', r2);
    assert.isTrue(r2 > 0.8);
  });
});



