const path = require("path");
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const createConfig = (target) => {
  return {
    mode: "production",
    devtool: "source-map",
    context: path.resolve(__dirname),
    entry: {
      // TODO: recover index entry
      logisticPredictor: './dist/model/linear-model/logistic-regression-predictor.js',
      logisticRegression: './dist/model/linear-model/logistic-regression.js'
    },
    target: target,
    output: {
      path: path.resolve(__dirname, "dist"),
      filename: '[name].bundle.js',
      library: "datacook"
    },
    plugins: [
      new BundleAnalyzerPlugin()
    ],
    module: {
      rules: [
        {
          use: {
            loader: "babel-loader",
            options: { presets: [ "@babel/preset-env" ] }
          },
          test: /\.(js|jsx)$/,
          exclude: /node_modules/
        }
      ]
    },
    resolve: {
      fallback: {
        fs: false
      }
    }
  };
};

module.exports = [
  createConfig("web")
];
