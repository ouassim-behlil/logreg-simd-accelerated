#include "include/LogisticRegression.hpp"
#include "include/logreg_dispatcher.hpp"
#include "include/simd_fn.hpp"
#include <cstring>
#include <cmath>

// Round n up to the next multiple of 8 (so every row is 32-byte aligned
// when stored as floats).
static inline int pad8(int n) { return (n + 7) & ~7; }

// -------------------------------------------------------------------
//  Construction / destruction
// -------------------------------------------------------------------

LogisticRegression::LogisticRegression(int n_features, float lr, int epochs)
    : n_features(n_features),
      padded_features(pad8(n_features)),
      lr(lr),
      epochs(epochs),
      bias(0.0f)
{
    weights = aligned_alloc_float(padded_features, 32);
    std::memset(weights, 0, padded_features * sizeof(float));
}

LogisticRegression::~LogisticRegression()
{
    aligned_free_float(weights);
}

// -------------------------------------------------------------------
//  Helper: copy row-major X [n_samples × n_features] into a padded,
//  32-byte-aligned buffer [n_samples × padded_features].
//  Extra columns are zero-filled so SIMD dot products are exact.
// -------------------------------------------------------------------
static float* copy_to_aligned(const float* X, int n_samples,
                               int n_features, int padded_features)
{
    const int pf = padded_features;
    float* buf = aligned_alloc_float((size_t)n_samples * pf, 32);
    if (!buf) return nullptr;

    for (int i = 0; i < n_samples; ++i) {
        std::memcpy(buf + (size_t)i * pf,
                    X   + (size_t)i * n_features,
                    n_features * sizeof(float));
        if (pf > n_features)
            std::memset(buf + (size_t)i * pf + n_features, 0,
                        (pf - n_features) * sizeof(float));
    }
    return buf;
}

// -------------------------------------------------------------------
//  Training – full-batch gradient descent
// -------------------------------------------------------------------

void LogisticRegression::train(const float* X, const int* Y, int n_samples)
{
    const int pf = padded_features;

    // 1) Copy all training data to an aligned, row-padded buffer.
    //    Each row starts on a 32-byte boundary so SIMD aligned
    //    loads are always safe.
    float* aligned_X = copy_to_aligned(X, n_samples, n_features, pf);

    // 2) Allocate work buffers (reused across epochs).
    float* z  = aligned_alloc_float(n_samples, 32);   // logits
    float* dw = aligned_alloc_float(pf, 32);           // weight gradient

    for (int epoch = 0; epoch < epochs; ++epoch) {

        // ---- forward pass: z_i = <w, x_i> + b ----
        for (int i = 0; i < n_samples; ++i) {
            z[i] = dot_product(aligned_X + (size_t)i * pf,
                               weights, pf) + bias;
        }

        // ---- sigmoid (SIMD-vectorised) ----
        float* p = sigmoid(z, n_samples);

        // ---- compute gradients ----
        std::memset(dw, 0, pf * sizeof(float));
        float db = 0.0f;

        for (int i = 0; i < n_samples; ++i) {
            float err = p[i] - static_cast<float>(Y[i]);
            db += err;
            const float* xi = aligned_X + (size_t)i * pf;
            for (int j = 0; j < n_features; ++j)
                dw[j] += err * xi[j];
        }

        // ---- parameter update ----
        const float inv_n = 1.0f / static_cast<float>(n_samples);
        for (int j = 0; j < n_features; ++j)
            weights[j] -= lr * inv_n * dw[j];
        bias -= lr * inv_n * db;

        aligned_free_float(p);
    }

    aligned_free_float(dw);
    aligned_free_float(z);
    aligned_free_float(aligned_X);
}

// -------------------------------------------------------------------
//  Single-sample prediction
// -------------------------------------------------------------------

float LogisticRegression::predict(const float* x) const
{
    const int pf = padded_features;

    // Copy into an aligned scratch buffer so the SIMD dot-product
    // can always use aligned loads.
    float* buf = aligned_alloc_float(pf, 32);
    std::memcpy(buf, x, n_features * sizeof(float));
    if (pf > n_features)
        std::memset(buf + n_features, 0,
                    (pf - n_features) * sizeof(float));

    float z = dot_product(buf, weights, pf) + bias;
    aligned_free_float(buf);

    return 1.0f / (1.0f + std::exp(-z));
}

int LogisticRegression::predict_class(const float* x) const
{
    return predict(x) >= 0.5f ? 1 : 0;
}

// -------------------------------------------------------------------
//  Batch prediction
// -------------------------------------------------------------------

void LogisticRegression::predict_batch(const float* X, float* out,
                                       int n_samples) const
{
    const int pf = padded_features;

    // Aligned copy of the whole input matrix.
    float* aligned_X = copy_to_aligned(X, n_samples, n_features, pf);

    // Compute logits into an aligned buffer.
    float* z = aligned_alloc_float(n_samples, 32);
    for (int i = 0; i < n_samples; ++i)
        z[i] = dot_product(aligned_X + (size_t)i * pf,
                           weights, pf) + bias;

    // Vectorised sigmoid.
    float* probs = sigmoid(z, n_samples);
    std::memcpy(out, probs, n_samples * sizeof(float));

    aligned_free_float(probs);
    aligned_free_float(z);
    aligned_free_float(aligned_X);
}

void LogisticRegression::predict_class_batch(const float* X, int* out,
                                             int n_samples) const
{
    float* probs = aligned_alloc_float(n_samples, 32);
    predict_batch(X, probs, n_samples);

    for (int i = 0; i < n_samples; ++i)
        out[i] = probs[i] >= 0.5f ? 1 : 0;

    aligned_free_float(probs);
}
