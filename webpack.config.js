const webpack = require("webpack");
const Stylish = require("webpack-stylish");
const path = require("path");

const ExtractTextPlugin = require("extract-text-webpack-plugin");

module.exports = {
  entry: ["babel-polyfill", "react-hot-loader/patch", "./src/main.js"],
  output: {
    path: __dirname + "/public/dist/",
    filename: "main.js",
    publicPath: "/dist"
  },
  node: {
    fs: 'empty'
  },
  module: {
    rules: [
      {
        test: /(\.js|\.jsx)$/,
        include: /src/,
        use: [{
          loader: 'babel-loader',
          options: {
            "cacheDirectory": true,
          }
        }]
      }, {
        test: /\.scss$/,
        use: ExtractTextPlugin.extract({
          fallback: 'style-loader',
          use: [
            'css-loader',
            'sass-loader'
          ]
        })
      }
    ]
  },
  watchOptions: {
    ignored: [/node_modules/]
  },
  devtool: "eval",
  devServer: {
    contentBase: "public/",
    hot: false,
    stats: "none"
  },
  plugins: [
    new webpack.NamedModulesPlugin(),
    new ExtractTextPlugin('css/style.css'),
    new Stylish()
  ]
};
