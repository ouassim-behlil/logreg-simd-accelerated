# ifndef LOG_REG_H
# define LOG_REG_H

# include <vector>

class LogisticRegression {
	public:
		LogisticRegression(int n_features);
		void	train(const float* X, const int* Y, int n_samples);
		float	predict(const float* x) const;

	private:
		int					n_features;
		std::vector<float> weights;
};

#endif