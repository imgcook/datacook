---
layout: default
title: twoSampleTTest()
parent: Statistics
nav_order: 1
lang: en
---

# twoSampleTTest()

The two-sample t-test is a method used to test whether the unknown population means of two groups are equal or not. It helps to answer questions like *"whether the average success rate is higher after implementing a new sales tool than before"* or *"whether the test results of patients who received a drug are better than test results of those who received a placebo"*. 

Therefore this method is wided used to analysze the result in A/B test. You can use this test when the values of data are independent and randomly sampled from two normal distributions.

## Import

```javascript
import * as datacook from '@pipcook/datacook';
const { twoSampleTTest } = datacook.Model; 
```

## Syntax

```typescript
twoSampleTTest(samples1: number[], samples2: number[]): TwoSampleTTestResult
```

## Parameters


| Parameter |        type        | description                                                         |
| :-------- | :-----------------: | :------------------------------------------------------------------ |
| samples1   | number[] | first sample data values |
| samples2    | number[] | second sample data values |

## Returns

 test result with following structure:
 ```typescript
 {
   t: number, /* t value for statistical test */
   pValue: number, /* p value */
   df: number, /* degrees of freedom */
   mean1: number, /* mean for first sample input */
   mean2: number, /* mean for second sample input */
   confidenceInterval: number, /* 95% confidence interval for x - y */
 }
 ```

## Usage

```javascript
const x = [5, 10, 6, 8, 9];
const y = [10, 8, 7, 6, 9];
twoSampleTTest(x, y);

/**
 * Two-Sample t-test
 * 
 * ┌─────────┬───────┬──────┬────────────────────┐
 * │ (index) │ Count │ Mean │ Standard Deviation │
 * ├─────────┼───────┼──────┼────────────────────┤
 * │    0    │   5   │ 7.6  │ 2.073644135332772  │
 * │    1    │   5   │  8   │ 1.5811388300841898 │
 * └─────────┴───────┴──────┴────────────────────┘
 * t = -0.34299717028501797
 * df = 8
 * p-value = 0.7404394537249616
 * 95 percent confidence interval:
 * [ -3.0892398363379705, 2.28923983633797 ]
 * **/
```




