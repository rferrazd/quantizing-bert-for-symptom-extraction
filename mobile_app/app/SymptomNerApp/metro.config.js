const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * `.onnx` is registered as an asset so the bundled model can be resolved at
 * runtime (via Image.resolveAssetSource → fetch → Uint8Array). Kept generic
 * (no version in the filename) so swapping V05 → V06 is a file replacement.
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    assetExts: [...defaultConfig.resolver.assetExts, 'onnx'],
  },
};

module.exports = mergeConfig(defaultConfig, config);
