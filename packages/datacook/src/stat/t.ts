
import { incbeta } from '../math/beta';
import * as Normal from './normal';
import { arrayMean1D, arrayVariance1D } from '../linalg/basic';
const cdfNormal = Normal.cdf;
/**
 * Cumulative density function of student distribution
 * @param x input number
 * @param df degree of freedom
 * @returns cdf of x in degree of freedom `df`
 */
export const cdf = (x: number, df: number): number => {
  const xSquare = x * x;
  const f = (x + Math.sqrt(xSquare + df)) / (2.0 * Math.sqrt(xSquare + df));
  if (df <= 100) {
    return incbeta(df / 2.0, df / 2.0, f);
  } else {
    /**
     * use the fact that when df close to infinity,
     * t distribution approximates to standard normal distribution
     **/
    return cdfNormal(x);
  }
};

const CRITICAL_VALUES_025 = [ 12.7062047361747, 4.30265272974946, 3.18244630528371, 2.77644510519779, 2.57058183563631, 2.44691185114497, 2.36462425159278, 2.30600413520417, 2.2621571627982, 2.22813885198627, 2.20098516009164, 2.17881282966723, 2.16036865646279, 2.1447866879178, 2.13144954555978, 2.11990529922125, 2.10981557783332, 2.10092204024104, 2.09302405440831, 2.08596344726586, 2.07961384472768, 2.07387306790403, 2.06865761041905, 2.06389856162803, 2.0595385527533, 2.05552943864287, 2.05183051648029, 2.04840714179524, 2.0452296421327, 2.04227245630124, 2.03951344639641, 2.0369333434601, 2.03451529744934, 2.03224450931772, 2.03010792825034, 2.02809400098045, 2.02619246302911, 2.02439416391197, 2.02269092003676, 2.02107539030627, 2.01954097044138, 2.01808170281844, 2.01669219922782, 2.01536757444376, 2.01410338888085, 2.01289559891943, 2.01174051372977, 2.01063475762423, 2.00957523712924, 2.00855911210076, 2.00758377031584, 2.00664680506169, 2.00574599531787, 2.00487928818806, 2.00404478328915, 2.00324071884787, 2.00246545929101, 2.00171748414524, 2.00099537808827, 2.00029782201426, 1.99962358499494, 1.99897151703338, 1.99834054252074, 1.99772965431769, 1.997137908392, 1.99656441895231, 1.9960083540253, 1.99546893142984, 1.99494541510724, 1.99443711177119, 1.99394336784563, 1.99346356666187, 1.99299712588985, 1.99254349518093, 1.99210215400224, 1.99167260964466, 1.99125439538838, 1.99084706881169, 1.99045021023013, 1.99006342125445, 1.9896863234569, 1.98931855713657, 1.98895978017516, 1.98860966697571, 1.98826790747722, 1.98793420623902, 1.98760828158907, 1.98728986483117, 1.98697869950628, 1.98667454070377, 1.98637715441862, 1.98608631695113, 1.98580181434582, 1.9855234418666, 1.9852510035055, 1.98498431152246, 1.98472318601398, 1.98446745450848, 1.98421695158642, 1.98397151852355 ];
const CRITICAL_VALUES_025_INF = 1.96;

export interface TwoSampleTTestResult {
  t: number,
  pValue: number,
  df: number,
  mean1: number,
  mean2: number,
  confidenceInterval: [number, number]
}


/**
 * Two sample t test. It is applied to compare whether the average difference between two groups is
 * really significant or if it is due instead to random chance. It helps to answer questions like
 * whether the average success rate is higher after implementing a new sales tool than before or
 * whether the test results of patients who received a drug are better than test results of those
 * who received a placebo.
 *
 * @param samples1 first sample input
 * @param samples2 second sample input
 * @returns test result with following structure:
 * {
 *  t: t value for statistical test,
 *  pValue: p value,
 *  df: degrees of freedom,
 *  mean1: mean for first sample input
 *  mean2: mean for second sample input
 *  confidenceInterval: 95% confidence interval for x - y
 * }
 */
export const twoSampleTTest = (samples1: number[], samples2: number[]): TwoSampleTTestResult => {
  const mean1 = arrayMean1D(samples1);
  const mean2 = arrayMean1D(samples2);
  const n1 = samples1.length;
  const n2 = samples2.length;
  const variance1 = arrayVariance1D(samples1, mean1);
  const variance2 = arrayVariance1D(samples2, mean2);
  const pulledVariance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2);
  const t = (mean1 - mean2) / (Math.sqrt(pulledVariance * (1 / n1 + 1 / n2)));
  const df = n1 + n2 - 2;
  const tcdf = cdf(t, df);
  const pValue = tcdf > 0.5 ? ((1 - tcdf) * 2) : (tcdf * 2);
  const tQuantile = df <= 100 ? CRITICAL_VALUES_025[df - 1] : CRITICAL_VALUES_025_INF;
  const sqrtTerm = Math.sqrt(variance1 / n1 + variance2 / n2);
  const confidenceInterval: [number, number] = [ (mean1 - mean2) - sqrtTerm * tQuantile, (mean1 - mean2) + sqrtTerm * tQuantile ];
  const statTable = [
    {
      Count: n1,
      Mean: mean1,
      'Standard Deviation': Math.sqrt(variance1)
    },
    {
      Count: n2,
      Mean: mean2,
      'Standard Deviation': Math.sqrt(variance2)
    }
  ];

  console.log('Two-Sample t-test\n');
  console.table(statTable);
  console.log(`t = ${t}\ndf = ${df}\np-value = ${pValue}`);
  // console.log('alternative hypothesis: true difference in means is not equal to 0');
  console.log('95 percent confidence interval:');
  console.log(confidenceInterval);
  return {
    t,
    pValue,
    df,
    mean1,
    mean2,
    confidenceInterval
  };
};
