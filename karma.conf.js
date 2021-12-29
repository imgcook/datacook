'use strict';

module.exports = function configureKarma(config) {
  config.set({
    browserDisconnectTimeout: 25000,
    browserNoActivityTimeout: 25000,

    // frameworks to use
    // available frameworks: https://npmjs.org/browse/keyword/karma-adapter
    frameworks: [
      'browserify',
      'mocha',
      'chai',
    ],

    files: [
      'dist/web/index.js',
      'test/browser/**/*.js'
    ],

    preprocessors: {
      'test/browser/**/*.js': [ 'browserify' ]
    },

    /**
     * Common options
     */
    singleRun: true,
    colors: true,
    logLevel: config.LOG_INFO,
    reporters: ['mocha'],

    // start these browsers
    // available browser launchers: https://npmjs.org/browse/keyword/karma-launcher
    browsers: [
      'ChromeHeadless',
    ],
  });
};
