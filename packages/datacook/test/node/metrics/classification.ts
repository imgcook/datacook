import { getConfusionMatrix, getPrecisionScore, getRecallScore, getF1Score, getF1Scores, getPrecisionScores, getRecallScores, getClassificationReport } from '../../../src/metrics/classifier';
import { assert } from 'chai';
import * as tf from '@tensorflow/tfjs-core';
import { numEqual } from '../../../src/math/utils';
import { tensorEqual } from '../../../src/linalg';


describe('Metrics', () => {
  it('get confusion metrics', async () => {
    const yTure = [ 0, 1, 2, 3 ];
    const yPred = [ 0, 1, 2, 3 ];
    const { confusionMatrix } = await getConfusionMatrix(yTure, yPred);
    confusionMatrix.print();
  });
  it('precision score', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const precisionScore = await getPrecisionScore(yTure, yPred);
    assert.isTrue(numEqual(precisionScore, 0.4, 1e-3));
    const precisionScoreMacro = await getPrecisionScore(yTure, yPred, 'macro');
    assert.isTrue(numEqual(precisionScoreMacro, 0.375, 1e-3));
    const precisionScoreWeighted = await getPrecisionScore(yTure, yPred, 'weighted');
    assert.isTrue(numEqual(precisionScoreWeighted, 0.3, 1e-3));
  });
  it('recall score', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const recallScore = await getRecallScore(yTure, yPred);
    assert.isTrue(numEqual(recallScore, 0.4, 1e-3));
    const recallScoreMacro = await getRecallScore(yTure, yPred, 'macro');
    assert.isTrue(numEqual(recallScoreMacro, 0.5, 1e-3));
    const recallScoreWeighted = await getRecallScore(yTure, yPred, 'weighted');
    assert.isTrue(numEqual(recallScoreWeighted, 0.4, 1e-3));
  });
  it('f1 score', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const f1Score = await getF1Score(yTure, yPred);
    assert.isTrue(numEqual(f1Score, 0.4, 1e-3));
    const f1ScoreMacro = await getF1Score(yTure, yPred, 'macro');
    assert.isTrue(numEqual(f1ScoreMacro, 0.416, 1e-3));
    const f1ScoreWeighted = await getF1Score(yTure, yPred, 'weighted');
    assert.isTrue(numEqual(f1ScoreWeighted, 0.333, 1e-3));
  });
  it('precision scores', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const { precisions } = await getPrecisionScores(yTure, yPred);
    assert.isTrue(tensorEqual(precisions, tf.tensor([ 1, 0.5, 0, 0 ])));
  });
  it('recall scores', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const { recalls } = await getRecallScores(yTure, yPred);
    assert.isTrue(tensorEqual(recalls, tf.tensor([ 1, 1, 0, 0 ])));
  });
  it('f1 scores', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const { f1s } = await getF1Scores(yTure, yPred);
    assert.isTrue(tensorEqual(f1s, tf.tensor([ 1, 0.667, 0, 0 ]), 1e-3));
  });
  it('classification report', async () => {
    const yTure = [ 1, 2, 3, 4, 3 ];
    const yPred = [ 1, 2, 4, 3, 2 ];
    const { f1s, precisions, recalls } = await getClassificationReport(yTure, yPred);
    assert.isTrue(tensorEqual(f1s, tf.tensor([ 1, 0.667, 0, 0 ]), 1e-3));
    assert.isTrue(tensorEqual(precisions, tf.tensor([ 1, 0.5, 0, 0 ]), 1e-3));
    assert.isTrue(tensorEqual(recalls, tf.tensor([ 1, 1, 0, 0 ]), 1e-3));
  });
});
