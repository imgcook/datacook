---
layout: default
title: Naive Bayes
parent: Models
lang: en
---

# Multinomial Naive Bayes

Naive Bayes model is a classical supervised classification algorithm based on Bayes' therom. Naive Bayes model applies the independence assumption of conditional probability for feature pairs.

$$
P(x_1,...,x_n|y) = P(x_1|y)P(x_2|y)...P(x_n|y)
$$

According to Bayes' therom, the posterior probabiliy of $y$ given $X$ is:

$$
P(y|x_1,..,x_n) = \frac{P(y)P(x_1,...,x_n|y)}{p(x_1,x_2...,x_n)}
$$

Multinomial Naive Bayes is a classical variant of naive bayes model, and is often used in text classification task. In this model, conditional probability $$P(x_i\|y)$$ is approximated by computing the frequency of $$x$$ in class $$y$$:

$$
\hat P(x_i|y) = \frac{N_{yi}+\alpha}{N_y + \alpha n}
$$

where $$N_{yi}$$ is the number of $$x_i$$ appeared in class $$y$$, $$n$$ is number of features, $\alpha$ is smooth parameter. Larger $\alpha$ often leads to more evenly results for all $$P(x_i\|y)$$.


## Import 

```typescript
import * as Datacook from 'datacook';
const { MultimonialNB } = DataCook.Model;
```

## Constructor

```typescript
const mnb = new MultimonialNB({ alpha: 0.1 });
```

### Option parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| alpha | number | smooth parameter, set it to a number greater than 0 to avoid overfitting and divide by zero error. larger number leads to more evenly result, **default = 1**|

## Methods

### fit

Training multinomial naive bayes model according to X, y.

#### Syntax
```typescript
async train(xData: Array<any> | Tensor, yData: Array<any> | Tensor): Promise<MultinomialNB>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray\<number\> | Tensor like of shape (n_samples, n_features), input feature |
| yData | Tensor \| Array\<any\> | Tensor like of shape (n_sample, ), input target values |

#### Returns

MultinomialNB


### predict 

Make predictions using naive bayes model.

#### Syntax

```typescript
async predict(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor | RecursiveArray<number> | Input features |

### predictProba

Predict probabilities for each class.

#### Syntax

```typescript
async predictProba(xData: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| xData | Tensor \| RecursiveArray<number> | Input features|

#### Returns

Predicted probabilities


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