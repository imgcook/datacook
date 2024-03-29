import { inverse } from '../../../src/linalg/inverse';
import * as tf from '@tensorflow/tfjs-core';
import { tensorEqual } from '../../../src/linalg/utils';
import { arrayMatmul2D, arrayTranpose2D } from '../../../src/linalg/basic';
import { assert } from 'chai';
import 'mocha';
const matrix = tf.tensor2d([
  [ 1, 2, 5, 10 ],
  [ 2, 6, 7, 5 ],
  [ 5, 7, 9, 6 ],
  [ 10, 5, 6, 7 ]
]);

const singularMatrix = tf.tensor2d([
  [ 1, 1, 3, 4 ],
  [ 2, 2, 3, 4 ],
  [ 3, 3, 6, 7 ],
  [ 4, 4, 7, 8 ]
]);

const originMatrix = [
  [
    1, 387, 5, 18,
    0, 37, 17, 27,
    59.42, 60, 21.661, 197,
    541, 30, 533, 386,
    277
  ],
  [
    1, 423, 3, 5,
    0, 6, 5, 0,
    57.509, 33, 13.983, 218,
    474, 23, 430, 422,
    236
  ],
  [
    1, 731, 12, 34,
    0, 14, 27, 0,
    61.021, 136, 28.274, 375,
    1018, 33, 831, 730,
    481
  ],
  [
    1, 757, 14, 33,
    0, 14, 29, 2,
    53.698, 113, 31.215, 303,
    909, 29, 679, 756,
    362
  ],
  [
    1, 417, 26, 15,
    0, 8, 8, 5,
    54.349, 51, 15.224, 195,
    663, 27, 624, 416,
    335
  ],
  [
    1, 322, 118, 585,
    0, 42, 197, 29,
    61.683, 33, 32.039, 112,
    283, 25, 227, 321,
    103
  ],
  [
    1, 361, 83, 1,
    0, 6, 0, 3,
    60.517, 77, 29.389, 113,
    453, 27, 485, 360,
    262
  ],
  [
    1, 579, 5, 32,
    0, 41, 32, 26,
    65.062, 66, 37.931, 188,
    530, 18, 400, 578,
    174
  ],
  [
    1, 416, 3, 66,
    0, 29, 63, 23,
    63.48, 66, 33, 192,
    541, 23, 470, 415,
    200
  ],
  [
    1, 394, 1, 11,
    0, 30, 8, 25,
    60.09, 97, 27.714, 169,
    728, 37, 705, 393,
    350
  ],
  [
    1, 600, 15, 0,
    1, 0, 0, 0,
    51.964, 112, 19.823, 286,
    1465, 32, 1149, 599,
    565
  ],
  [
    1, 150, 2, 0,
    1, 0, 0, 0,
    65.329, 22, 27.16, 65,
    164, 16, 149, 149,
    81
  ],
  [
    1, 168, 5, 0,
    1, 0, 0, 0,
    65.55, 51, 30.723, 119,
    364, 21, 313, 168,
    166
  ],
  [
    1, 725, 11, 0,
    1, 0, 0, 0,
    62.711, 137, 19.826, 308,
    1372, 24, 1160, 724,
    691
  ],
  [
    1, 28, 2, 0,
    1, 0, 0, 0,
    75.556, 7, 43.75, 26,
    38, 12, 43, 27,
    16
  ],
  [
    1, 77, 9, 0,
    1, 0, 0, 0,
    63.59, 16, 30.769, 61,
    107, 27, 110, 76,
    52
  ],
  [
    1, 323, 18, 0,
    1, 0, 0, 0,
    60.313, 86, 49.425, 214,
    554, 25, 454, 322,
    174
  ],
  [
    1, 234, 9, 0,
    1, 0, 0, 0,
    64.572, 38, 20.994, 144,
    350, 20, 304, 233,
    181
  ],
  [
    1, 156, 17, 0,
    1, 0, 0, 0,
    68.009, 20, 30.303, 69,
    149, 16, 108, 155,
    66
  ],
  [
    1, 391, 8, 0,
    1, 0, 0, 0,
    65.689, 62, 36.257, 159,
    501, 28, 459, 390,
    171
  ],
  [
    1, 13, 0, 0,
    0, 1, 0, 1,
    87.473, 4, 66.667, 12,
    21, 3, 9, 13,
    6
  ],
  [
    1, 78, 22, 0,
    1, 0, 0, 0,
    88.205, 1, 100, 38,
    50, 2, 2, 78,
    1
  ],
  [
    1, 37, 18, 0,
    0, 0, 0, 0,
    75.118, 6, 42.857, 16,
    28, 11, 24, 36,
    14
  ],
  [
    1, 68, 7, 1,
    0, 2, 0, 0,
    67.324, 10, 71.429, 33,
    71, 21, 66, 67,
    14
  ],
  [
    1, 185, 84, 0,
    1, 0, 0, 0,
    82.973, 1, 100, 88,
    175, 4, 17, 184,
    1
  ],
  [
    1, 26, 1, 2,
    0, 0, 2, 0,
    80.474, 4, 40, 13,
    21, 6, 23, 25,
    10
  ],
  [
    1, 18, 0, 0,
    0, 0, 0, 0,
    78.858, 4, 80, 10,
    20, 7, 9, 17,
    5
  ],
  [
    1, 8, 0, 0,
    0, 1, 0, 0,
    87.867, 2, 66.667, 2,
    4, 5, 5, 7,
    3
  ],
  [
    1, 13, 0, 0,
    0, 1, 0, 1,
    85.691, 1, 50, 4,
    4, 4, 4, 13,
    2
  ],
  [
    1, 66, 5, 0,
    0, 2, 0, 1,
    70.794, 13, 65, 24,
    61, 17, 40, 65,
    20
  ],
  [
    1, 2, 1, 0, 0, 0, 0,
    0, 100, 1, 0, 0, 0, 0,
    0, 2, 0
  ],
  [
    1, 150, 2, 7,
    0, 1, 0, 1,
    63.028, 26, 20.8, 103,
    242, 24, 202, 149,
    125
  ],
  [
    1, 261, 2, 3,
    0, 3, 2, 0,
    63.928, 57, 31.148, 135,
    385, 27, 368, 260,
    183
  ],
  [
    1, 64, 0, 18,
    0, 8, 18, 5,
    68.492, 8, 28.571, 27,
    53, 15, 56, 63,
    28
  ],
  [
    1, 3, 0, 0, 0, 1, 0,
    1, 100, 1, 0, 1, 1, 0,
    0, 3, 0
  ],
  [
    1, 56, 0, 1,
    0, 0, 0, 0,
    75.106, 14, 35.897, 33,
    63, 16, 65, 55,
    39
  ],
  [
    1, 240, 2, 34,
    0, 33, 30, 23,
    64.717, 64, 36.994, 131,
    362, 27, 302, 240,
    173
  ],
  [
    1, 206, 5, 265,
    0, 74, 259, 47,
    64.253, 31, 21.233, 124,
    338, 29, 333, 205,
    146
  ],
  [
    1, 2, 1, 0, 0, 0, 0,
    0, 100, 1, 0, 0, 0, 0,
    0, 2, 0
  ],
  [
    1, 36, 0, 4,
    0, 19, 4, 17,
    63.727, 7, 20.588, 39,
    71, 13, 79, 34,
    34
  ],
  [
    1, 1052, 0, 1234,
    0, 850, 1234, 761,
    60.292, 99, 11.099, 202,
    1473, 13, 1174, 1052,
    892
  ],
  [
    1, 2137, 29, 2570,
    0, 1467, 2570, 1420,
    40.292, 79, 4.994, 544,
    2887, 27, 2723, 2135,
    1582
  ],
  [
    1, 1614, 24, 1766,
    0, 959, 1714, 901,
    44.314, 172, 14.286, 436,
    2196, 28, 2202, 1614,
    1204
  ],
  [
    1, 1466, 34, 1685,
    0, 962, 1663, 901,
    51.254, 108, 9.774, 457,
    2026, 25, 1972, 1464,
    1105
  ],
  [
    1, 2964, 132, 359,
    0, 744, 318, 593,
    50.404, 433, 24.842, 883,
    4519, 50, 4304, 2963,
    1743
  ],
  [
    1, 1574, 39, 1832,
    0, 1026, 1831, 989,
    46.802, 84, 7.311, 484,
    2098, 31, 2069, 1573,
    1149
  ],
  [
    1, 1552, 24, 1981,
    0, 1087, 1978, 1051,
    42.823, 44, 3.833, 500,
    2149, 23, 1909, 1550,
    1148
  ],
  [
    1, 1470, 31, 1684,
    0, 963, 1668, 909,
    52.128, 105, 9.477, 479,
    1987, 23, 1994, 1468,
    1108
  ],
  [
    1, 1325, 34, 1532,
    0, 885, 1527, 855,
    41.083, 81, 7.918, 495,
    1932, 24, 1722, 1323,
    1023
  ],
  [
    1, 2843, 406, 2490,
    0, 1480, 2490, 1473,
    4.329, 1, 0.062, 1231,
    3174, 5, 2394, 2841,
    1624
  ],
  [
    1, 19, 0, 0,
    0, 1, 0, 0,
    58.042, 1, 5, 11,
    38, 3, 7, 18,
    20
  ],
  [
    1, 845, 24, 0,
    0, 1, 0, 0,
    29.64, 2, 0.612, 250,
    657, 9, 654, 844,
    327
  ],
  [
    1, 221, 6, 0,
    0, 0, 0, 0,
    48.851, 14, 9.091, 129,
    328, 22, 296, 220,
    154
  ],
  [
    1, 77, 0, 0,
    0, 0, 0, 0,
    46.525, 1, 1.818, 57,
    91, 3, 74, 76,
    55
  ],
  [
    1, 40, 0, 0,
    0, 0, 0, 0,
    60.133, 1, 5.556, 18,
    36, 3, 3, 39,
    18
  ],
  [
    1, 124, 3, 0,
    0, 0, 0, 0,
    41.926, 1, 1.563, 74,
    120, 9, 124, 123,
    64
  ],
  [
    1, 100, 3, 7,
    0, 5, 7, 5,
    56.636, 7, 7.778, 114,
    177, 23, 140, 99,
    90
  ],
  [
    1, 200, 15, 0,
    0, 0, 0, 0,
    34.194, 1, 0.806, 99,
    244, 6, 232, 199,
    124
  ],
  [
    1, 81, 9, 2,
    0, 0, 2, 0,
    52.852, 1, 3.704, 29,
    52, 6, 52, 80,
    27
  ],
  [
    1, 135, 3, 67,
    0, 6, 67, 6,
    39.606, 1, 1.25, 77,
    153, 7, 130, 134,
    80
  ],
  [
    1, 2120, 492, 0,
    1, 0, 0, 0,
    32.995, 1, 0.99, 178,
    1594, 2, 101, 2119,
    101
  ],
  [
    1, 880, 37, 47,
    0, 320, 47, 309,
    51.021, 194, 25.228, 326,
    1674, 32, 1752, 879,
    769
  ],
  [
    1, 1860, 196, 444,
    0, 290, 420, 241,
    60.245, 284, 22.885, 574,
    2592, 45, 2644, 1860,
    1241
  ],
  [
    1, 997, 17, 515,
    0, 123, 418, 94,
    49.959, 109, 12.358, 408,
    2253, 25, 1901, 997,
    882
  ],
  [
    1, 995, 29, 3,
    0, 75, 3, 22,
    61.56, 234, 28.502, 581,
    1926, 32, 1517, 994,
    821
  ],
  [
    1, 1047, 78, 30,
    0, 46, 26, 18,
    57.089, 183, 31.552, 383,
    1365, 37, 1111, 1046,
    580
  ],
  [
    1, 980, 79, 16,
    0, 165, 11, 134,
    58.023, 265, 39.97, 346,
    1717, 31, 1396, 979,
    663
  ],
  [
    1, 2178, 110, 2387,
    0, 349, 2379, 200,
    46.901, 358, 16.776, 819,
    4319, 50, 4316, 2175,
    2134
  ],
  [
    1, 970, 48, 0,
    1, 0, 0, 0,
    56.593, 185, 28.997, 448,
    1656, 39, 1329, 969,
    638
  ],
  [
    1, 1597, 4, 3,
    0, 15, 1, 1,
    8.197, 1, 0.063, 2872,
    3166, 2, 1584, 1596,
    1583
  ]
];

const quadra = arrayMatmul2D(arrayTranpose2D(originMatrix), originMatrix);

describe('Matrix Solver', () => {

  it('get inverse matrix', async () => {
    const invM = await inverse(matrix);
    const iM = tf.matMul(matrix, invM);
    const isIdMatrix = tensorEqual(iM, tf.eye(iM.shape[0]), 1e-2);
    assert.isTrue(isIdMatrix);
  });

  it('solve singular matrix throw type error', async () => {
    let err: Error;
    try {
      await inverse(singularMatrix);
    } catch (error) {
      err = error;
    }
    assert.isTrue(err && err instanceof TypeError);
  });

  it('solve a real matrix', async () => {
    try {
      await inverse(tf.tensor(quadra));
    } catch (error) {
      console.log(error);
    }
  });
});
