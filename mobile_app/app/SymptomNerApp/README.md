# Symptom NER — iOS app

Single-screen React Native app that runs a BioBERT NER model **entirely on-device** to extract affirmed (POS) / denied (NEG) symptoms from clinical notes.

This README covers how to run the app on the iOS Simulator. For deployment to a physical device / TestFlight / App Store, see [`../../DEPLOY_IOS.md`](../../DEPLOY_IOS.md).

---

## Day-to-day: I just closed Metro, how do I get back to running?

Two terminals, both inside this folder (`mobile_app/app/SymptomNerApp`).

**Terminal A — start Metro (the JS dev server):**
```sh
npm start
```
Leave it running. If it complains `EADDRINUSE: address already in use :::8081`, kill the stale process: `lsof -ti:8081 | xargs kill -9`, then `npm start` again.

**Terminal B — choose one:**
- **App is already installed on the simulator** (from a previous `npm run ios`): just open Simulator.app and tap the **Symptom NER** icon. It auto-connects to Metro.
- **App is NOT installed yet, or you want a fresh build:**
  ```sh
  npm run ios
  ```
  This builds, installs, and launches in the simulator. First build after a clean is slow (~5 min) because of the 77 CocoaPods including ONNX Runtime; subsequent rebuilds are fast.

After it's up, JS edits hot-reload automatically. To force a reload: press `r` in Metro, or `Cmd+R` in the simulator.

---

## First-time setup (fresh clone / new machine)

You only need these once.

### 1. Toolchain prerequisites
- **Xcode** (any recent version). After install, point the command-line tools at it:
  ```sh
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  ```
- **iOS Simulator runtime** — Xcode 26+ no longer bundles it; download once:
  ```sh
  xcodebuild -downloadPlatform iOS
  ```
  (~8 GB, takes a while.)
- **CocoaPods** via Homebrew Ruby (system Ruby on macOS is too old):
  ```sh
  brew install ruby
  /opt/homebrew/lib/ruby/gems/4.0.0/bin/gem install cocoapods
  ```
- **Node.js** — any LTS.

### 2. Install JS dependencies
```sh
cd mobile_app/app/SymptomNerApp
npm install
```

### 3. Bundle the model assets into the app
The 108 MB ONNX model is **not** in git — it's regenerated from `mobile_app/artifacts/<version>/` via a script. From the repo root:
```sh
python mobile_app/model_prep/bundle_app_assets.py --version v05 --quant int8
```
This writes `assets/model/{model.onnx, vocab.json, config.json}` inside the app. Re-run with a different `--version` to switch model versions (e.g. v06 later).

### 4. Install iOS native deps (CocoaPods)
```sh
cd ios
LANG=en_US.UTF-8 /opt/homebrew/lib/ruby/gems/4.0.0/bin/pod install
cd ..
```
The `LANG=en_US.UTF-8` prefix is required when calling `pod` from a non-interactive shell — without it CocoaPods crashes on Unicode.

### 5. First build
```sh
npm start            # in one terminal
npm run ios          # in another
```

That's it. From here on, the day-to-day flow above is all you need.

---

## When do I need to do what?

| Change | What to re-run |
|---|---|
| JS / TS code edit | Nothing — Fast Refresh handles it. Press `r` in Metro if needed. |
| New JS dependency (`npm install some-pkg`) | Restart Metro: `Ctrl+C` then `npm start` again. JS hot-reloads. |
| New **native** dependency (anything that requires a pod) | `pod install` in `ios/`, then full `npm run ios` rebuild. |
| Model swap (e.g. v05 → v06) | Re-run `bundle_app_assets.py` with the new `--version`, then `npm run ios`. Also restart Metro with `npm start -- --reset-cache` to avoid asset caching surprises. |
| `metro.config.js` change | Restart Metro with `--reset-cache`. |
| Tokenizer / inference logic (Block C code) | JS edit — Fast Refresh, plus `npm test` to keep the parity gates green. |

---

## Running the test suite
```sh
npm test
```
This runs the tokenizer parity gate (token IDs, word alignment, char offsets vs. Python `AutoTokenizer`) and the aggregation/span gate (vs. Python `inference_utils.py`). Both must stay green — they're what guarantee the on-device output matches the training/eval pipeline.

---

## Troubleshooting

- **`error iOS devices or simulators not detected`** → no Simulator runtime is installed. Run `xcodebuild -downloadPlatform iOS`.
- **App shows the default React Native welcome screen** → you're looking at a browser at `localhost:8081`, not the simulator. Find the **Simulator** window (the phone-shaped one) and look there. If the simulator itself shows the welcome, force a reload (`Cmd+R`) or restart Metro with `--reset-cache`.
- **`VirtualizedLists should never be nested...`** error → should not happen; the symptom list was refactored out of a nested ScrollView. If it appears, something regressed in `App.tsx`.
- **Pod install fails on Unicode** → you forgot the `LANG=en_US.UTF-8` prefix.
- **Build error about `onnxruntime-react-native`** → re-run `pod install` after `npm install` — the native module needs both.
