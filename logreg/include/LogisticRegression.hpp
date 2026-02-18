#ifndef LOG_REG_H
# define LOG_REG_H

# include <cstdint>

// ---------------------------------------------------------------
//  LogisticRegression
//  Binary classifier trained with full-batch gradient descent.
//  All internal buffers are 32-byte aligned so AVX loads never
//  split a cache line.  The dispatcher (init_kernels) must be
//  called before constructing this object.
// ---------------------------------------------------------------
class LogisticRegression {
public:
	// n_features : number of input features (excluding bias)
	// lr         : learning rate  (default 0.1)
	// epochs     : full passes over the training set (default 1000)
	LogisticRegression(int n_features,
	                   float lr     = 0.1f,
	                   int   epochs = 1000);

	~LogisticRegression();

	LogisticRegression(const LogisticRegression&)            = delete;
	LogisticRegression& operator=(const LogisticRegression&) = delete;

	// Train on a flat row-major matrix X  [n_samples × n_features]
	// and integer labels Y [n_samples] ∈ {0, 1}.
	void	train(const float* X, const int* Y, int n_samples);

	// Returns P(y=1 | x)  ∈ (0, 1)  for a single sample.
	float	predict(const float* x) const;

	// Returns the predicted class label (0 or 1) for a single sample.
	int		predict_class(const float* x) const;

	// Batch prediction: write P(y=1|x_i) into out[0..n_samples-1].
	void	predict_batch(const float* X, float* out, int n_samples) const;

	// Batch classification: write 0/1 into out[0..n_samples-1].
	void	predict_class_batch(const float* X, int* out, int n_samples) const;

	int		get_n_features() const { return n_features; }

private:
	int		n_features;
	int		padded_features;   // n_features rounded up to next multiple of 8
	float	lr;
	int		epochs;

	float*	weights;           // 32-byte aligned, length = padded_features
	float	bias;
};

#endif