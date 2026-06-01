/**
 * session.ts
 *
 * Loads the bundled ONNX model and creates a singleton ONNX Runtime inference
 * session for on-device NER. The model is shipped as a Metro asset (.onnx) and
 * read into memory via fetch → Uint8Array, which avoids any extra native
 * filesystem dependency and works in both dev and release.
 *
 * Input/output tensor names are read from the loaded session at runtime rather
 * than hardcoded, so a different export (e.g. V06, or a model with/without
 * token_type_ids) does not require code changes here.
 *
 * Memory note: the model (~104 MB INT8) is briefly held as a JS Uint8Array
 * during session creation. For a single load on a modern device this is fine;
 * if memory becomes a problem (very large models / low-end devices), switch to
 * a file-path load via a native FS module and stream from disk instead.
 */

import { Image } from 'react-native';
import { InferenceSession } from 'onnxruntime-react-native';

// Generic filename (no version) — see bundle_app_assets.py / metro.config.js.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const MODEL_ASSET = require('../../assets/model/model.onnx');

/** Metadata about a loaded session: the runtime tensor names. */
export type SessionInfo = {
  session: InferenceSession;
  inputNames: string[];
  outputNames: string[];
};

let sessionInfoPromise: Promise<SessionInfo> | null = null;

/**
 * Resolves the bundled model asset to a Uint8Array of its bytes. Works for both
 * the Metro dev server (http asset URI) and release builds (file:// URI).
 */
async function loadModelBytes(): Promise<Uint8Array> {
  const source = Image.resolveAssetSource(MODEL_ASSET);
  if (!source?.uri) {
    throw new Error('Could not resolve model.onnx asset URI');
  }
  const response = await fetch(source.uri);
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

/**
 * Creates (once) and returns the ONNX inference session plus its input/output
 * tensor names. Subsequent calls return the same cached promise.
 */
export function getSession(): Promise<SessionInfo> {
  if (sessionInfoPromise === null) {
    sessionInfoPromise = (async () => {
      const bytes = await loadModelBytes();
      const session = await InferenceSession.create(bytes);
      return {
        session,
        inputNames: session.inputNames as string[],
        outputNames: session.outputNames as string[],
      };
    })().catch(err => {
      // Don't cache the failure — clear the slot so the next call can retry,
      // otherwise a transient first-load failure would brick the app until restart.
      sessionInfoPromise = null;
      throw err;
    });
  }
  return sessionInfoPromise;
}
