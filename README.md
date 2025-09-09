# PillCounter — Auto‑tuned Pill Instance Counter

This project aims at developing a robust, parameter‑light computer vision pipeline that **counts pills in arbitrary images** by fusing **luminance** and **chroma** cues, generating reliable **watershed seeds**, and returning per‑pill **instance masks, bounding boxes, and counts**. The codebase is modular (header/implementation separation) and compilable from the CLI.

---

## TL;DR — Build & Run

**Prerequisites**

* OpenCV 4.x (with image codecs; macOS via Homebrew: `brew install opencv`)
* Clang++ with C++17 (Apple Clang or LLVM)
* Sample images under `images/`

**Compile**

```bash
clang++ -std=c++17 src/main.cpp src/pill_counter.cpp -o bin/pc_app `pkg-config --cflags --libs opencv4`
```

**Execute (single image)**

```bash
./bin/pc_app images/blue-pills-white-bg.jpg
```

**What you’ll see**

* Console: Number of pills counted
* Windows: Intermediate masks and final boxes
* Optional: Files written to `results/`

---

## Project Structure

```
.
├── include/
│   └── pill_counter.hpp     # Public API (declarations & defaults)
├── src/
│   ├── pill_counter.cpp     # Implementations (no default args)
│   └── main.cpp             # Thin CLI demo / batch runner
├── images/                  # Input images
│   ├── blue-pills-white-bg.jpg     
│   └── red-pills-white-bg.jpg  
├── results/                 # result overlays & masks
└── bin/                     # Built executable
```

### Header: `pill_counter.hpp`

* Declares the **namespace** (e.g., `pilseg`) and the public API:

  * `FusionOutputs fusedMaskForWatershed(...)`
  * `WSOut runWatershed(...)`
  * Optional helpers: `luminanceMask(...)`, `chromaMask(...)`, etc.
* **Default parameters** live here (not in `.cpp`) to avoid ODR issues.
* Keeps `main.cpp` concise and readable.

### Implementation: `pill_counter.cpp`

* Implements declared functions (no defaults; identical signatures).
* Internal helpers:

  * `autoClipLimit`, `autoBlockSize`
  * `chromaMagnitudeF32`, `normalizeToU8`
  * `makeSeedsFromFused`, `watershedSeeds`
* Enforces **type/size safety** before watershed:

  * Source coerced to **BGR 8UC3**
  * Masks **aligned to image resolution**
  * Markers are **CV\_32S**

### App: `main.cpp`

* Loads an image (or iterates a folder)
* Calls `fusedMaskForWatershed(...)` → `runWatershed(...)`
* Draws boxes, prints counts, optionally saves artifacts

---

## Pill Counter CV Pipeline — End‑to‑End

1. **Color Stabilization (BGR → Lab)**
   One conversion reused by both luminance and chroma to ensure consistent statistics.

2. **Luminance Mask (Shadow‑Tolerant)**

   * Extract **L\*** (lightness).
   * Apply **CLAHE** with **auto clip limit** from L\* contrast (`autoClipLimit`).
   * Threshold via:

     * **Adaptive Gaussian** (default) for uneven light/soft shadows, or
     * **Otsu** (global) for uniform scenes.
   * Morphological **CLOSE** to heal gaps.

3. **Chroma Mask (Color Magnitude)**

   * Compute **C = sqrt(a² + b²)** from Lab a\*, b\*.
   * Segment chroma using **Otsu** (global) or **K‑means (k=2)**.
   * Morphological **OPEN** to remove specks.

4. **Fusion**

   * `fused = luminanceMask ∧ chromaMask`
     Gates out bright backgrounds and soft shadows while retaining color‑rich pills.

5. **Seed Generation (Auto)**

   * Distance transform on `fused`.
   * **Percentile cut** (e.g., 65th) for sure‑foreground—no magic numbers.
   * Sure‑background via **dilation**; unknown region = bg − fg.
   * **Markers** for watershed: `0 = unknown`, `1 = background`, `>=2 = objects`.

6. **Watershed Segmentation**

   * Run `cv::watershed` with aligned sizes/types.
   * Convert labels to a binary instance mask; light morphology to polish boundaries.

7. **Object Extraction & Counting**

   * `connectedComponentsWithStats` yields blobs and **bounding boxes**.
   * Area floor is **relative to image area** (auto), not a fixed constant.
   * Count = number of boxes (optionally post‑process with NMS to de‑dup overlaps).

---

## Entirely Auto-Tuned Setup

* **Auto thresholds everywhere**:

  * CLAHE **clip limit** from L\* std‑dev.
  * Adaptive threshold **block size** scales with image size.
  * Adaptive **C** offset scales with contrast.
  * **Otsu** adapts globally; **K‑means** adapts per‑image distribution.
  * Seed **percentile** replaces fixed distances.
  * Area filter is **image‑size aware**.
* **Single Lab conversion** prevents drift between luminance and chroma.
* **Type/size alignment** eliminates resizing bugs that force hard‑coded patches.

Result: robust across backgrounds, shadows, and pill colors—**no per‑image tuning**.

---

## Usage Patterns

### Single Image

```bash
./bin/pc_app images/blue-pills-white-bg.jpg
```

### Batch Mode (optional)

If your `main.cpp` iterates `images/` with `std::filesystem`, just drop images in and run:

```bash
./bin/pc_app
```

It will log counts and (optionally) write overlays to `out/`.

---

## Troubleshooting

* **`imwrite`: could not find a writer for the specified extension**
  Your OpenCV lacks PNG/JPEG codecs.

  * Save `.bmp` instead, or
  * Reinstall OpenCV with codecs (e.g., Homebrew OpenCV), or
  * Implement a fallback saver (PNG → JPG → BMP).

* **`watershed`: assertion `src.size() == dst.size()` failed**
  Mismatch between image size and markers.

  * Ensure image is **CV\_8UC3**; markers are **CV\_32S**; both have the **same size**.
  * The provided `runWatershed` enforces these constraints.

* **`std::clamp` errors**
  Use C++17 (`-std=c++17`) or replace with `std::max(lo, std::min(hi, v))`.

* **Apple Silicon**
  Ensure OpenCV and your build are the same architecture; use `pkg-config --cflags --libs opencv4`.

---

## Extensibility

* Swap chroma segmentation: pass `"otsu"` or `"kmeans"` to `chromaMask`.
* Tighten/loosen instances: adjust `fgPercentile` in `runWatershed`.
* Control specks/smoothing: tweak morphological kernels (3–5 px) and relative area floor.
