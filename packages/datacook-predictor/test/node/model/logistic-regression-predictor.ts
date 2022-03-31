import { assert } from 'chai';
import 'mocha';
import * as DataCook from '../../../src';
import { LogisticRegressionPredictor } from '../../../src/model/linear-model/logistic-regression-predictor';
import { accuracyScore } from '../../../src/metrics/classification';
import { Matrix } from 'ml-matrix';

const weight: number[] = [ 2, 3, 1, -4, 6 ];
const cases: number[][] = [];

const nData = 2000;
for (let i = 0; i < nData; i++) {
  cases.push([ Math.random()*100, Math.random()*100, Math.random()*200, Math.random()*100, Math.random()*10 ]);
}
const eta = cases.map((c) => {
  return  c[0] * weight[0] + c[1] * weight[1] + c[2] * weight[2] + c[3] * weight[3] + c[4] * weight[4] - 20;
})
const y = eta.map((e) => (e > 0) ? 1 : 0);
const yMult = eta.map((e: number): number => e < -200 ? 1 : e < 0 ? 2 : e < 200 ? 3 : 4);

const modelJson = '{"name":"LogisticRegression","classes":[0,1],"fitIntercept":true,"penalty":"none","c":1,"optimizerType":"adam","optimizerProps":{"learningRate":0.1},"modelWeights":[[[0.7305909991264343],[1.3003342151641846],[0.3077864646911621],[-1.3392314910888672],[2.3634321689605713]],[-5.018787860870361]],"featureSize":5,"outputSize":1}';
const modelJsonMult = '{"name":"LogisticRegression","classes":[1,2,3,4],"fitIntercept":true,"penalty":"none","c":1,"optimizerType":"adam","optimizerProps":{"learningRate":0.1},"modelWeights":[[[-0.6940170526504517,-0.2643401324748993,0.25622645020484924,0.39840662479400635],[-0.8198472261428833,-0.42843952775001526,0.367847204208374,0.727942943572998],[-0.35221490263938904,-0.10543955117464066,0.036920759826898575,0.184328094124794],[1.2087854146957397,0.5236815214157104,-0.45859482884407043,-0.8881247043609619],[-1.5070888996124268,-0.6187031865119934,1.0892512798309326,1.7055954933166504]],[-12.140129089355469,13.451814651489258,9.924636840820312,-12.331857681274414]],"featureSize":5,"outputSize":4}';

describe('Logistic Predictor', () => {
  it('load model as predictor', async () => {
    const lr = new LogisticRegressionPredictor();
    await lr.fromJson(modelJson);
    const predY = await lr.predict(cases);
    const acc = accuracyScore(y, predY);
    console.log('accuracy: ', acc);
    assert.isTrue(acc >= 0.75);
  });
  it('load model as predictor and calculate probability', async () => {
    const lr = new LogisticRegressionPredictor();
    await lr.fromJson(modelJson);
    const predY = await lr.predict(cases);
    const predProba = await lr.predictProba(cases);
    console.log(predProba);
    //predProba.reduce((s: bool, p: number, i: number): boolean => i == 0 ? p <= 1 : p <=1 && s);
    const acc = accuracyScore(y, predY);
    console.log('accuracy: ', acc);
    assert.isTrue(acc >= 0.75);
  });
  it('load model as predictor (multi-classification)', async () => {
    const lr = new LogisticRegressionPredictor();
    await lr.fromJson(modelJsonMult);
    const predY = await lr.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy: ', acc);
    assert.isTrue(acc >= 0.75);
  });
  it('load model as predictor and calculate probability (multi-classification)', async () => {
    const lr = new LogisticRegressionPredictor();
    await lr.fromJson(modelJsonMult);
    const predY = await lr.predict(cases);
    const acc = accuracyScore(yMult, predY);
    console.log('accuracy: ', acc);
    assert.isTrue(acc >= 0.75);
  });
});
