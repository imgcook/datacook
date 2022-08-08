const karmaTypescriptConfig = {
  tsconfig: 'tsconfig.test.json',
  // Disable coverage reports and instrumentation by default for tests
  coverageOptions: { instrumentation: false },
  reports: {},
  bundlerOptions: {
    sourceMap: false
  }
};

module.exports = function(config) {
  config.set({
    frameworks: [ "mocha", "karma-typescript" ],
    karmaTypescriptConfig,
    files: [
      { pattern: 'test/setup.ts', type: 'js' },
      { pattern: "src/**/*.ts", type: 'js' },
      { pattern: "src/*.ts", type: 'js' },
      { pattern: "test/**/*.ts", type: 'js' }
    ],
    preprocessors: {
      "test/setup.ts": [ "karma-typescript" ],
      "src/**/*.ts": [ "karma-typescript" ],
      "test/**/*.ts": [ "karma-typescript" ]
    },
    reporters: [ "dots", "karma-typescript" ],

    browsers: [ "ChromeHeadless" ],

    singleRun: true
  });
};
