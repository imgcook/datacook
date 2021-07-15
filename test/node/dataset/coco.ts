import { expect } from 'chai';
import { makeDatasetFromCocoFormat } from '../../../src/dataset/coco';
import 'mocha';
import * as fs from 'fs-extra';
import * as sinon from 'sinon';

const cocoAnnFileData = {

};

// todo(feely): `sinon` not works for now
describe('Coco Dataset', () => {
  it('should make a dataset from coco format', async () => {
    // const readJsonStub = sinon.stub(fs, 'readJson').resolves(cocoAnnFileData);
    // console.log('meta', fs.readJson);
    // const dataset = makeDatasetFromCocoFormat({
    //   trainDir: '/trainDir',
    //   trainAnnotationFile: 'trainAnnotationFile.json',
    //   testDir: '/testDir',
    //   testAnnotationFile: 'testAnnotationFile.json',
    //   validDir: '/validDir',
    //   validAnnotationFile: 'validAnnotationFile.json'
    // });

    // expect(await dataset.getDatasetMeta()).to.eql(meta);
    // expect(await dataset.train.next()).to.eql(sample);
    // expect(await dataset.test.next()).to.eql(sample);
    // expect(await dataset.train.nextBatch(3)).to.eql([sample, sample]);
    // expect(await dataset.test.nextBatch(1)).to.eql([sample]);
  });
});
