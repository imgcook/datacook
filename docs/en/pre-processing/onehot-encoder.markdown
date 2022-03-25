---
layout: default
title: OneHotEncoder
parent: Pre-processing
nav_order: 1
lang: en
---

# OnehotEncoder

OneHotEncoder is used to encode categorical features as a one-hot numeric array.

## Import

```javascript
import * as datacook from '@pipcook/datacook';
const { OneHotEncoder } = datacook.Preprocess;
```

## Constructor

```javascript
const onehotEncoder = new OnehotEncoder({ drop: "first" });
```

### Option in Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| drop `optional` | \{ 'none', 'binary-only', 'first' \} | Specifies a method to drop one of the categories per feature, which is useful to avoid collinear problem. However, dropping one category may introduce a bias term in downstream models.<br/>**'none'**: default, return all features <br/>**'first'**: drop the first categories in each feature<br/>**'binary-only'**: drop the first category in each feature with two categories.<br/> **default='none'**|


## Properties

### categories `<Tensor1D>`

One dimensional tensor of categories. Onehot result will be consistent with the order of categories appeared in this array.

### drop `<'first' | 'binary-only' | 'none'>`

Drop method for this encoder.

## Methods

### init()

Initialize one-hot encoder

#### Syntax

```javascript
init(x: Tensor1D | number[] | string[]): Promise<void>
```

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor1D \| number[] \| string[] | data input used to initialize encoder | 

#### Example

```javascript
const onehotEncoder = new OneHotEncoder();
await onehotEncoder.init([ '1', '2', '3' ]);
```

### encode()

Encode a given feature into one-hot format

#### Syntax

```javascript
async encode(x: Tensor | number[] | string[]): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| number[] \| string[] | original data need to encode |


#### Returns

`<Tensor>` transformed one-hot feature

#### Example

```javascript
const onehotEncoder = new OneHotEncoder();
await onehotEncoder.init([ '1', '2', '3' ]);
const encoded = await onehotEncoder.encode(['3', '3', '2']);
/**
 * Tensor
 *   [[0, 0, 1],
 *    [0, 0, 1],
 *    [0, 1, 0]]
 * /
```


### decode()

Decode one-hot array to original category array

#### Syntax

```javascript
async decode(x: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| RecursiveArray<number> | one-hot format data need to transform |

#### Returns

`<Tensor>` transformed category data

#### Example

```javascript
const onehotEncoder = new OneHotEncoder();
await onehotEncoder.init([ '1', '2', '3' ]);
const encoded = await onehotEncoder.decode([
    [0, 0, 1], [0, 0, 1], [0, 1, 0]
]);
/**
 * Tensor
 *   ['1', '2', '3']
 * /
```











