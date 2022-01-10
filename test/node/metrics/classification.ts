import { generateClassificationReport } from "../../../src/metrics/classifier";
import '@tensorflow/tfjs-backend-cpu';

const yTure = [ 'A', 'A', 'B', 'D', 'E', 'B', 'E', 'B', 'D', 'D' ];
const yPred = [ 'A', 'B', 'A', 'E', 'E', 'B', 'E', 'B', 'A', 'D' ];
describe('Classification Report', () => {
  it('get report', async () => {
    const report = await generateClassificationReport(yTure, yPred);
    report.classes.print();
    report.confusionMatrix.print();
    console.log('f1s');
    report.f1s.print();
    console.log('precisions');
    report.precisions.print();
    console.log('recalls');
    report.recalls.print();
  });
});
