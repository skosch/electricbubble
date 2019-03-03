const webpack = require("webpack");
const Stylish = require("webpack-stylish");

module.exports = {
  entry: "./src/main.js",
  output: {
    path: __dirname + "/public/dist/",
    filename: "main.js",
    publicPath: "/dist"
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: "babel-loader"
      }
    ]
  },
  plugins: [new Stylish()],
  devtool: "source-map",
  devServer: {
    contentBase: "public/"
  }
};
