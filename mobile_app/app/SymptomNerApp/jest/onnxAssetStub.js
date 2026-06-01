/**
 * onnxAssetStub.js
 *
 * Stand-in for `.onnx` asset imports in the Jest environment, where Metro's
 * binary-asset bundling isn't available. Returns a dummy asset id so requires
 * resolve without loading the real (~104 MB) model.
 */
module.exports = 1;
