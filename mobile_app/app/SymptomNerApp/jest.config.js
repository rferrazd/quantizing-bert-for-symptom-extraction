/**
 * jest.config.js
 *
 * Jest configuration. In addition to the React Native preset, this:
 *  - maps `.onnx` asset imports to a stub (Jest can't bundle binary assets), and
 *  - mocks `onnxruntime-react-native`, whose native JSI binding is unavailable
 *    in the Node test environment, so component tests that import the app don't
 *    crash at module load. Pure-logic tests (e.g. the tokenizer) are unaffected.
 */

module.exports = {
  preset: '@react-native/jest-preset',
  moduleNameMapper: {
    '\\.onnx$': '<rootDir>/jest/onnxAssetStub.js',
  },
  setupFiles: ['<rootDir>/jest/setup.js'],
};
