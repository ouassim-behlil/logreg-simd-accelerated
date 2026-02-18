# SIMD Logistic Regression

A binary logistic regression classifier written in C++ from scratch, using SIMD intrinsics for speed and pybind11 for NumPy integration.

## What I learned

- **How logistic regression works internally** — forward pass (dot product + sigmoid), computing gradients, and updating weights with gradient descent.
- **SIMD intrinsics (SSE, AVX, AVX2+FMA)** — using `_mm_load_ps`, `_mm256_load_ps`, `_mm256_fmadd_ps` etc. to process 4 or 8 floats at once instead of one at a time.
- **Runtime CPU detection** — using `cpuid` to check what the CPU supports, then dispatching to the best kernel at runtime through function pointers.
- **Memory alignment** — SIMD aligned loads (`_mm256_load_ps`) require 32-byte aligned pointers. Used `posix_memalign` and padded feature vectors to multiples of 8 so every row stays aligned.
- **Approximating exp/sigmoid with SIMD** — implemented a Horner-scheme polynomial approximation of `exp()` entirely in SIMD registers, then built `sigmoid(x) = 1/(1+exp(-x))` on top of it.
- **pybind11 + NumPy** — wrapping a C++ class so it can be called from Python with NumPy arrays. Used `forcecast` to handle any dtype/layout NumPy throws at it.

## Build

### C++ only (Linux / macOS)

```bash
make
./main
```

### C++ only (Windows / cross-platform via CMake)

```bash
cmake -B build
cmake --build build --config Release
./build/main          # Linux / macOS
build\Release\main    # Windows
```

### Python module (Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pybind11 numpy
make python
```

### Python module (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pybind11 numpy
pip install -e .
```

Or on any platform:

```bash
pip install -e .
```

## Usage

### C++ example

```cpp
#include "logreg/include/LogisticRegression.hpp"
#include "logreg/include/logreg_dispatcher.hpp"

int main() {
    init_kernels();

    float X[] = {1, 3,  2, 4,  3, 1,  4, 2};
    int   Y[] = {1, 1, 0, 0};

    LogisticRegression model(2, 0.1f, 1000);
    model.train(X, Y, 4);

    float sample[] = {1.5f, 3.5f};
    int cls = model.predict_class(sample);  // → 1
}
```

### Python example

```python
import numpy as np
import logreg

X = np.array([[1, 3], [2, 4], [3, 1], [4, 2]], dtype=np.float32)
Y = np.array([1, 1, 0, 0], dtype=np.int32)

model = logreg.LogisticRegression(n_features=2, lr=0.1, epochs=1000)
model.train(X, Y)

# single prediction
prob = model.predict(X[0])       # P(y=1) ≈ 0.99
cls  = model.predict_class(X[0]) # 1

# batch prediction
probs   = model.predict_batch(X)       # array of probabilities
classes = model.predict_class_batch(X)  # array of 0s and 1s
```

## Requirements

- **Compiler:** g++, clang++, or MSVC with C++17 and x86 SIMD support
- **Build:** Make (Linux/macOS) or CMake 3.15+ (any platform)
- **Python (optional):** 3.8+, pybind11, numpy
