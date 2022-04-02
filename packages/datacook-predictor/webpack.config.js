const path = require("path");
const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const createConfig = (target) => {
  return {
    mode: "production",
    devtool: "source-map",
    context: path.resolve(__dirname),
    entry: {
      index: './src/index.ts',
      kmeansPredictor: './src/model/clustering/kmeans-predictor.ts'
    },
    target: target,
    output: {
      path: path.resolve(__dirname, "dist"),
      filename: '[name].bundle.js',
      chunkFilename: '[id].js',
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
        },
        {
          test: /\.tsx?$/,
          // ts-loader是官方提供的处理tsx的文件
          use: 'ts-loader',
          exclude: /node_modules/
        }
      ]
    },
    resolve: {
      extensions: [ '.tsx', '.ts', '.js' ],
      fallback: {
        fs: false
      }
    },
    optimization: {
      sideEffects: false,
      usedExports: true
    }
  };
};

module.exports = [
  createConfig("web")
];
