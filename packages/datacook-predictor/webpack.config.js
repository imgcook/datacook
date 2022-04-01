const path = require("path");
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const createConfig = (target) => {
  return {
    mode: "production",
    devtool: "source-map",
    context: path.resolve(__dirname),
    entry: {
      // TODO: recover index entry
      kmeansPredictor: './dist/model/clustering/kmeans-predictor'
    },
    target: target,
    output: {
      path: path.resolve(__dirname, "dist"),
      filename: '[name].bundle.js',
      library: "datacook"
    },
    plugins: [
      new CleanWebpackPlugin(),
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
