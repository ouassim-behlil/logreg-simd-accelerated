#!/usr/bin/env python3
"""
Quick smoke-test for the SIMD logistic-regression extension.

Build first with either:
    make python           # Makefile shortcut
    pip install -e .      # setuptools / editable install
"""

import numpy as np
import logreg

# ------------------------------------------------------------------
#  Generate a linearly-separable dataset
# ------------------------------------------------------------------
rng = np.random.default_rng(42)
n_samples, n_features = 500, 4

X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
true_w = np.array([1.5, -2.0, 0.8, -1.2], dtype=np.float32)
z = X @ true_w + 0.5
Y = (z > 0).astype(np.int32)

# 80 / 20 split
X_train, X_test = X[:400], X[400:]
Y_train, Y_test = Y[:400], Y[400:]

# ------------------------------------------------------------------
#  Train
# ------------------------------------------------------------------
model = logreg.LogisticRegression(n_features=n_features, lr=0.05, epochs=500)
print(f"Model: n_features={model.n_features}, training on {len(X_train)} samples …")
model.train(X_train, Y_train)

# ------------------------------------------------------------------
#  Evaluate
# ------------------------------------------------------------------

# Batch prediction
probs   = model.predict_batch(X_test)
classes = model.predict_class_batch(X_test)
accuracy = np.mean(classes == Y_test) * 100
print(f"Test accuracy (batch) : {accuracy:.1f}%")

# Single-sample prediction
prob = model.predict(X_test[0])
cls  = model.predict_class(X_test[0])
print(f"Single sample  → P(y=1) = {prob:.4f}, predicted = {cls}, true = {Y_test[0]}")

# ------------------------------------------------------------------
#  Verify alignment is handled for sliced / non-contiguous arrays
# ------------------------------------------------------------------
X_sliced = X_test[::2]          # non-contiguous view → forcecast copies
preds = model.predict_class_batch(X_sliced)
print(f"Sliced array   → predicted {len(preds)} samples OK")

# integer labels in int64 (numpy default) – forcecast converts to int32
Y_i64 = Y_train.astype(np.int64)
model2 = logreg.LogisticRegression(n_features=n_features, lr=0.05, epochs=100)
model2.train(X_train, Y_i64)     # should not raise
print("int64 labels   → accepted OK")

print("\nAll checks passed ✓")
