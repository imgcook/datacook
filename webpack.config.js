'use strict';

const path = require('path');
const webpack = require('webpack');

function configure(target) {
  return {
    devtool: 'source-map',
    context: path.resolve(__dirname),
    entry() {
      return ([
        'index',
        'model/linear-model/logistic-regression',
        'model/linear-model/logistic-regression-predictor',
      ]).reduce(function appendEntrySource(entry, name) {
        const importPath = path.join(__dirname, 'src', `${name}.ts`);
        entry[name] = importPath;
        return entry;
      }, {});
    },
    target,
    output: {
      clean: true,
      path: path.resolve(__dirname, 'dist', target),
      filename: '[name].js',
      library: {
        name: 'datacook',
        type: 'assign-properties',
      },
    },
    plugins: [
      new webpack.ProvidePlugin({
        Buffer: ['buffer', 'Buffer'],
        process: 'process/browser',
      }),
    ],
    module: {
      rules: [
        {
          use: 'ts-loader',
          test: /\.ts$/,
          exclude: [
            /node_modules/,
          ],
        }
      ]
    },
    resolve: {
      extensions: ['.ts', '.js'],
      fallback: {
        path: require.resolve('path-browserify'),
        zlib: require.resolve('browserify-zlib'),
        http: require.resolve('stream-http'),
        https: require.resolve('https-browserify'),
        stream: require.resolve('stream-browserify'),
        buffer: require.resolve('buffer'),
        fs: false,
      },
    },
    experiments: {
      asyncWebAssembly: true,
    },
  };
}

module.exports = [
  configure('web'),
];
