/**
 * setup.js
 *
 * Jest setup. Mocks `onnxruntime-react-native` so importing the app under test
 * doesn't initialize the native JSI binding (absent in Node). The mock provides
 * a minimal InferenceSession whose create() resolves to a stub session; tests
 * that need real inference behaviour should override this per-test.
 */

jest.mock('onnxruntime-react-native', () => ({
  InferenceSession: {
    create: jest.fn(async () => ({
      inputNames: ['input_ids', 'attention_mask', 'token_type_ids'],
      outputNames: ['logits'],
      run: jest.fn(async () => ({})),
    })),
  },
  Tensor: jest.fn(),
}));
