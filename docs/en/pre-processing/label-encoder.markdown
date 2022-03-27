---
layout: default
title: LabelEncoder
parent: Pre-processing
nav_order: 1
lang: en
---

# LabelEncoder

LabelEncoder is used to transform categorical array into array of numerical indices.

## Import 

```javascript
import * as datacook from '@pipcook/datacook';
const { LabelEncoder } = datacook.Preprocess;
```

## Constructor

```javascript
const labelEncoder = new LabelEncoder({drop: 'first'});
```

### Option in Prameters 


| parameter | type | description |
| --------- | ---- | ----------- |
| drop `optional` | \{ 'none', 'binary-only', 'first' \} | Specifies a method to drop one of the categories per feature.<br/>**'none'**: default, return all features <br/>**'first'**: drop the first categories in each feature<br/>**'binary-only'**: drop the first category in each feature with two categories.<br/> **default='none'**|



## Properties

### categories `<Tensor1D>`

One dimensional tensor of categories. Encoded result will be consistent with the order of categories appeared in this array.

### drop `<'first' | 'binary-only' | 'none'>`

Drop method for this encoder.

## Methods

### init()

Initialize label encoder

#### Syntax

```javascript
init(x: Tensor1D | number[] | string[]): Promise<void>
```

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor1D \| number[] \| string[] | data input used to initialize encoder | 

#### Example

```javascript
const labelEncoder = new LabelEncoder();
await labelEncoder.init([ '1', '2', '3' ]);
```

### encode()

Encode a given feature into numberic format

#### Syntax

```javascript
async encode(x: Tensor | number[] | string[]): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| number[] \| string[] | original data need to encode |


#### Returns

`<Tensor>` transformed feature array

#### Example

```javascript
const labelEncoder = new LabelEncoder();
await labelEncoder.init([ '1', '2', '3' ]);
const encoded = await labelEncoder.encode(['3', '3', '2']);
/**
 * Tensor
 *    [2, 2, 1]
 * /
```


### decode()

Decode numeric array to original category array

#### Syntax

```javascript
async decode(x: Tensor | RecursiveArray<number>): Promise<Tensor>
```

#### Parameters

| parameter | type | description |
| --------- | ---- | ----------- |
| x | Tensor \| number[] | numeric format data need to transform |

#### Returns

`<Tensor>` transformed category data

#### Example

```javascript
 const labelEncoder = new LabelEncoder();
 await labelEncoder.init([ '1', '2', '3' ]);
 const decoded = await labelEncoder.decode([ 1, 2, 2, 1, 0]);
 decoded.print();
/**
 * Tensor
 *    ['2', '3', '3', '2', '1']
 * /
```



