---
layout: default
title: Nonlinear Regression
parent: Models
lang: en
---

# NonLinearRegression

Nonlinear regression is a form of regression analysis in which observational data are modeled by a function
which is a nonlinear combination of independent variables.

$$ y \sim f(X, \beta) $$

## Import 

```typescript
import * as Datacook from 'datacook';
const { NonLinearRegression } = DataCook.Model;
```


## Constructor

```typescript
const nlr = new NonLinearRegression();
```

## Properties

### coeffs <Tensor[]>

Fitting variables object.

## Methods

### fit

Fitting nonlinear regression model according to X, y.

#### Syntax
```typescript
async fit(expr: string | ((tf: any, features: Tensor<Rank>, ...args: Variable[]) => Scalar),
    x: Tensor | RecursiveArray<number>,
    y: Tensor | RecursiveArray<number>,
    params: NonLinearRegressionTrainParams): Promise<void>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| expr | string \| ((tf: any, features: Tensor<Rank>, ...args: Variable[]) => Scalar) | Expression of relationship between x and y, should be a function or string of function. Parameters of this functions: `tf`: first parameter, tfjs object, can be used to represent the numberic relationship defined in tfjs; `features`: input x; `args`: fitting variables. Return of this function should be a tf.Scalar. exp: `(tf, x, a, b) => tf.add(tf.mul(x, a), b)` |
| xData | Tensor \| RecursiveArray\<number\> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| Array\<any\> | Tensor like of shape (n_sample, ), input target values |
| params | {initPrams: number[], tol?: number, maxIterTimes?: number} | Fitting paramters, `initParams`: init parameters, should be a numeric array and given by the order of variables defined in `expr`. `tol`: tolerence of convergence, **default=1e-4**, `maxIterTimes`: maximum iterate times for approximation, **default=100**|

#### Example

```typescript
const nlr = new NonLinearRegression();
await nlr.fit((tf, x, a, b) => tf.mul(tf.add(x, a), b), x, y, { initParams: [1, 2] })
```

### predict

Make prediction using fitted model

#### Syntax

```typescript
async predict(x: Tensor): Promise<Tensor>
```

#### Parameters


| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Input features |

### fromJson

Load model paramters from json string object


#### Syntax

```typescript
async fromJson(modelJson: string)
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| modelJson | string | model json string |


### toJson

Dump model parameters to json string.

#### Syntax

```typescript
async toJson(): Promise<string>
```

#### Returns

string of model json

## Examples

Following is an example of fitting an exponential relationship between `x` and `y`.

```html
<div id="chart"></div>
<script>
    const { Mix } = G2Plot;
    async function fit() {
        const { NonLinearRegression } = datacook.Model;
        const nlr = new NonLinearRegression();
        const x = Array.from(new Array(50).keys()).map((d) => [d]);
        const y = x.map((d) => 5 * Math.exp(-0.5 * d[0]) + Math.random() - 0.5);
        console.log(x);
        console.log(y);
        await nlr.fit((tf, x, a, b) => tf.squeeze(tf.mul(b, tf.exp(tf.mul(tf.neg(a), x)))),
            x,
            y,
            {initParams: [0.4, 5]});
        console.log(nlr);
        const a = nlr.coeffs[0].dataSync()[0];
        const b = nlr.coeffs[1].dataSync()[0];
        console.log(a,b);
        const data = x.map((d) => ({
            x: d[0],
            yPred: b * Math.exp(-a * d[0])
        }));
        const originData = x.map((d, i) => ({
            x: d[0],
            yTrue: y[i]
        }));
        console.log(data);
        console.log(originData);
        const chart = new Mix('chart', {
            appendPadding: 8,
            syncViewPadding: true,
            tooltip: { shared: true, showMarkers: false, showCrosshairs: true, offsetY: -50 },
            views: [
            {
                data: originData,
                axes: {},
                meta: {
                    yTrue: {
                        sync: true
                    }
                },
                geometries: [
                    {
                        type: 'point',
                        xField: 'x',
                        yField: 'yTrue',
                        mapping: {
                            shape: 'circle',
                            style: {
                            fillOpacity: 1,
                            },
                        }
                    },
                ],
            },
            {
                data: data,
                axes: false,
                meta: {
                    yPred: {
                        sync: 'yTrue',
                    },
                },
                geometries: [
                    {
                        type: 'line',
                        xField: 'x',
                        yField: 'yPred',
                        mapping: {},
                    },
                ],
            }]
        });
        chart.render();
    }
    fit();
</script>
```

<div id="chart"></div>
<script>
    const { Mix } = G2Plot;
    async function fit() {
        const { NonLinearRegression } = datacook.Model;
        const nlr = new NonLinearRegression();
        const x = Array.from(new Array(50).keys()).map((d) => [d]);
        const y = x.map((d) => 5 * Math.exp(-0.5 * d[0]) + Math.random() - 0.5);
        console.log(x);
        console.log(y);
        await nlr.fit((tf, x, a, b) => tf.squeeze(tf.mul(b, tf.exp(tf.mul(tf.neg(a), x)))),
            x,
            y,
            {initParams: [0.4, 5]});
        console.log(nlr);
        const a = nlr.coeffs[0].dataSync()[0];
        const b = nlr.coeffs[1].dataSync()[0];
        console.log(a,b);
        const data = x.map((d) => ({
            x: d[0],
            yPred: b * Math.exp(-a * d[0])
        }));
        const originData = x.map((d, i) => ({
            x: d[0],
            yTrue: y[i]
        }));
        console.log(data);
        console.log(originData);
        const chart = new Mix('chart', {
            appendPadding: 8,
            syncViewPadding: true,
            tooltip: { shared: true, showMarkers: false, showCrosshairs: true, offsetY: -50 },
            views: [
            {
                data: originData,
                axes: {},
                meta: {
                    yTrue: {
                        sync: true
                    }
                },
                geometries: [
                    {
                        type: 'point',
                        xField: 'x',
                        yField: 'yTrue',
                        mapping: {
                            shape: 'circle',
                            style: {
                            fillOpacity: 1,
                            },
                        }
                    },
                ],
            },
            {
                data: data,
                axes: false,
                meta: {
                    yPred: {
                        sync: 'yTrue',
                    },
                },
                geometries: [
                    {
                        type: 'line',
                        xField: 'x',
                        yField: 'yPred',
                        mapping: {},
                    },
                ],
            }]
        });
        chart.render();
    }
    fit();
</script>