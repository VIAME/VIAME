var webpack = require("webpack");

module.exports = {
  devServer: {
    port: 8081,
    public: process.env.PUBLIC_ADDRESS,
    proxy: {
      "/api": {
        target: process.env.API_PROXY,
        secure: false
      }
    }
  },
  publicPath: process.env.VUE_APP_STATIC_PATH,
  chainWebpack: config => {
    config.resolve.symlinks(false);
    config.module
      .rule("additional")
      .test(/\.m?jsx?$/)
      .include.add(/@girder\/components/)
      .add(/vue-utilities/)
      .end()
      .use()
      .loader("babel-loader");
  },
  configureWebpack: () => {
    return {
      plugins: [
        new webpack.DefinePlugin({
          "process.env": {
            VERSION: JSON.stringify(require("./package.json").version)
          }
        })
      ]
    };
  }
};
