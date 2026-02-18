#include "logreg/include/LogisticRegression.hpp"
#include "logreg/include/logreg_dispatcher.hpp"
#include "logreg/include/simd_fn.hpp"
#include <iostream>
#include <cstring>

int main()
{
    // Detect and select the best SIMD kernels for this CPU.
    init_kernels();

    // ---- tiny linearly-separable 2-D dataset ----
    //   class 1: points above the line y = x
    //   class 0: points below the line y = x
    constexpr int N = 8;
    constexpr int F = 2;

    float X[N * F] = {
        1.0f, 3.0f,   // 1
        2.0f, 4.0f,   // 1
        3.0f, 5.0f,   // 1
        0.5f, 2.5f,   // 1
        3.0f, 1.0f,   // 0
        4.0f, 2.0f,   // 0
        5.0f, 3.0f,   // 0
        2.5f, 0.5f,   // 0
    };
    int Y[N] = {1, 1, 1, 1, 0, 0, 0, 0};

    LogisticRegression model(F, /*lr=*/0.1f, /*epochs=*/1000);
    model.train(X, Y, N);

    // ---- evaluate on the training set ----
    std::cout << "\n--- Predictions ---\n";
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        float prob = model.predict(X + i * F);
        int   cls  = model.predict_class(X + i * F);
        std::cout << "Sample " << i
                  << ":  P(y=1) = " << prob
                  << "   class = " << cls
                  << "   (true = " << Y[i] << ")\n";
        if (cls == Y[i]) ++correct;
    }
    std::cout << "\nAccuracy: " << correct << " / " << N << "\n";

    // ---- batch prediction ----
    float probs[N];
    int   classes[N];
    model.predict_batch(X, probs, N);
    model.predict_class_batch(X, classes, N);

    std::cout << "\n--- Batch predictions ---\n";
    for (int i = 0; i < N; ++i)
        std::cout << "Sample " << i
                  << ":  prob = " << probs[i]
                  << "   class = " << classes[i] << "\n";

    return 0;
}
