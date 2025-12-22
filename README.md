# PersonalFinanceAnalysis
To manage my personal expenses


Z-score: Measures how far an observation deviates from the mean. It assumes that the data follows a normal distribution.
$$z_i = \frac{x_i - \overline{x}}{\sigma}$$
where $\sigma$ is the standard deviation. Since it relies on the mean and stddev, it is quite susceptible to outliers. Therefore, we use the modified Z-score:

$$mz_i = 0.6745 \,\frac{x_i-\tilde{x}}{MAD}$$

Where $\tilde{x}$ is the median and MAD the median absolute deviation value. An observation is considered a potential outlier if it falls more than 3.5 MAD from the median, i.e. $\vert mz_i \vert > 3.5$.
