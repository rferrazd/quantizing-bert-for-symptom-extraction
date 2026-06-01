# iOS Deployment Plan

Plan for taking the Symptom NER app off the simulator and onto real devices, including (optionally) the App Store. Written 2026-05-31.

## Current state
The app runs on the iOS Simulator only, via `npm run ios`. This document covers physical-device install, TestFlight distribution, and App Store submission — in increasing order of effort and risk.

---

## Prerequisites

- **Apple Developer Program** — **$99 / year**. Required for non-simulator signing, TestFlight, and App Store submission. Enrollment takes 1–2 business days for Apple to approve.
- **Free alternative**: a regular Apple ID can sign builds for your *own* physical devices, but the signing expires every 7 days. Fine for personal poking, useless for distribution.

---

## 1. One-time setup

1. Enroll at [developer.apple.com](https://developer.apple.com)
2. In Xcode → Settings → Accounts: add the Apple ID
3. The current Bundle ID is the React Native template default: `org.reactjs.native.example.SymptomNerApp`. **This must change** to something globally unique, e.g. `com.<yourdomain>.symptomner`, and be registered as an App ID at developer.apple.com → Certificates, Identifiers & Profiles
4. In [App Store Connect](https://appstoreconnect.apple.com): create the app record (Bundle ID, name, primary language)

---

## 2. Project changes required

| File | Change |
|---|---|
| `ios/SymptomNerApp.xcodeproj/project.pbxproj` | `PRODUCT_BUNDLE_IDENTIFIER` → new unique ID |
| `ios/SymptomNerApp/Info.plist` | `CFBundleDisplayName` (the name shown under the icon) |
| `ios/SymptomNerApp/Info.plist` | Bump `CFBundleShortVersionString` (e.g. 1.0.0) and `CFBundleVersion` for every upload |
| `ios/SymptomNerApp/Info.plist` | `NSMicrophoneUsageDescription` — only if STT (`@react-native-voice/voice`) gets wired in |

Code signing is managed in Xcode → SymptomNerApp target → Signing & Capabilities. With "Automatically manage signing" on and an enrolled Team selected, Xcode creates certs/profiles for you. Manual signing is only worth it if you need CI/CD reproducibility.

---

## 3. Build for distribution

1. In Xcode, switch scheme to **Release**: Product → Scheme → Edit Scheme → Run → Build Configuration: Release
2. Select destination **"Any iOS Device (arm64)"**
3. Product → **Archive** (the build takes ~20–40 min first time due to all 77 pods + bundling the 108 MB ONNX model)
4. When done, the **Organizer** window opens — your archive appears there. Click **Distribute App** → App Store Connect → Upload

---

## 4. Distribution path A — TestFlight (recommended first)

The lowest-risk way to put it on a real iPhone.

- Upload from Organizer; the build processes in App Store Connect (~10–30 min)
- Add testers:
  - **Internal** (up to 100): App Store Connect users on your team — instant access, **no App Review**
  - **External** (up to 10,000): anyone with an email — requires a quick TestFlight review (usually <1 day)
- Testers install via the TestFlight iOS app
- Builds expire every 90 days — re-upload to refresh
- **Free** beyond the $99/year developer fee

This is enough for a learning project that just wants to show "I built a thing that runs on a real iPhone."

---

## 5. Distribution path B — App Store submission

Much more effort, and **non-trivial risk of rejection** for this specific app. Read "Review risks" below before committing time here.

Required metadata in App Store Connect:
- App description (4000 char max), keywords (100 char), category
- **Privacy policy URL** — mandatory, even though no data leaves the device
- **Support URL** — can be a GitHub issues page
- Screenshots: minimum iPhone 6.7" (1290×2796) and 6.5" (1242×2688). Generated easily from the simulator via Device → Screenshot
- Age rating questionnaire
- **App Review notes**: explicitly tell the reviewer this is an educational ML demo with no medical advice, all inference on-device

Review typically takes 1–3 business days. First-attempt rejection is common for borderline apps.

---

## Review risks specific to this app

These are real and worth pricing in before spending time on App Store submission:

- **Medical-app guidelines** (App Store Guideline 1.4.1, 5.1.1(iii)): the app analyzes "patient" notes and outputs symptom labels. Apple's reviewers treat anything that *looks* clinical with high scrutiny. Mitigations:
  - Position as **Education / Developer Tools** category, *not* Medical
  - Lead the description with "Educational demonstration of an on-device NLP model — not a diagnostic tool"
  - Keep the in-app "For learning purposes only" disclaimer prominent (already done)
  - Privacy policy must explicitly state no medical advice / no PHI handling
- **Model bundle size** (108 MB): well within current limits (no longer a hard cap since iOS 13), but expect questions. Justify in review notes: "Model runs entirely on-device to avoid any user data leaving the phone."
- **Privacy nutrition label**: declare **"Data Not Collected"** — easy win since inference is local
- **Export compliance**: BioBERT is a published model, no custom cryptography → answer **"No"** to the encryption question (or declare exempt under 5D002)

---

## Pricing recommendation

For a learning project: **Free, no in-app purchases.**

Reasoning:
- Easiest review (no commercial / tax / banking forms)
- No App Store commission to think about
- Aligns with the project's stated goal (learning, not commercial product)
- Avoids the App Store paid-app review bar, which is higher

If you ever wanted to monetize:
- Lowest paid tier is **$0.99** (Tier 1)
- Apple takes **30%** by default, or **15%** if enrolled in the [Small Business Program](https://developer.apple.com/app-store/small-business-program/) (businesses earning <$1M/yr)
- A learning project is unlikely to clear that paid-review bar without a clearer commercial story than "ML demo"

---

## Costs summary

| Item | Cost |
|---|---|
| Apple Developer Program (mandatory) | $99 / year |
| App Store fees (free app) | $0 |
| Certs, profiles, TestFlight | $0 (included) |
| App Store commission (paid apps only) | 30% (or 15% via Small Business Program) |
| Domain for privacy policy URL (optional) | $0 (a GitHub Pages site works) |

**Bottom line: $99/year covers everything for a free learning app.**

---

## Realistic timeline

| Phase | Effort | Calendar time |
|---|---|---|
| Apple Developer enrollment | 30 min | 1–2 days for approval |
| Bundle ID, signing, version setup | 1 hour | same day |
| First Release archive + upload | 1 hour | ~30 min build + ~30 min processing |
| On a real iPhone via TestFlight | — | same day once processed |
| App Store metadata + screenshots + privacy policy | 2–4 hours | 1 day |
| App Store submission → approval | 30 min | 1–3 days review (expect at least one rejection on a medical-adjacent app) |

---

## Recommended path

1. **Enroll Apple Developer account** — required for everything below
2. **Change the Bundle ID and sign with the new Team** — needs to happen exactly once
3. **Get a TestFlight build on your physical iPhone** — this is the "I shipped it" moment for a learning project. Free, no review friction.
4. **Use the app yourself for a few weeks** — surface real issues that don't appear in the simulator (perf, thermals, the 2× peak memory during model load)
5. **Only then** decide whether App Store submission is worth pursuing, or whether TestFlight (with the 90-day re-upload cycle) is sufficient indefinitely

---

## Out of scope here

- Android / Google Play — has its own process; deferred per project decision (iOS first)
- CI/CD for signing (Fastlane, EAS, etc.) — only worth it once you're shipping multiple builds per week
- App icon / splash design — tracked separately in the app-cover work
