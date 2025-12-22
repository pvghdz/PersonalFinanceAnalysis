# PersonalFinanceAnalysis
Bunch of scripts to manage my personal finances

## 1. Determining a budget. An exercise in outlier detection

My first idea was: take all your data and compute a few means and stddevs and use them to define a rule or warning for each expense category; e.g. "your monthly expenses in food are now on the 90th percentile of what you spend on food each month", or "your travel costs are annormally high (defined as above the mean + 2 stddevs)".

However, this is more complicated than initially though. For starters, since I began recoding how much I spend, I have moved twice and gotten different jobs, which means that my budgets have adjusted. Simply using normalized data (e.g., food/income instead of only food) is not good since the historiacal variability on how much I spend is too high: my recurring costs in Saarbücken, which amounter to ca. 270€/month for rent ate 1/3 of my budget, are not comparable to those in München since now I share a much higher rent with my girlfriend. I also have to pay for my own insurances (which used to be included in my scolarship), electricity, gas, internet (which were all included in my SB rent before), and my Deutschland Ticket (which used to come with my student ID). The expenses are simply not comparable.

So, I decided to let the program examine only the data corresponding to my moving into München.

The second bigger issue arises: what do I do with the months where I incurred extraordinary expenses? E.g., when I paid the deposit for my flat or when I bought a new computer. My solution is to take out the outliers from the dataset and use the rest of the observations to get a bunch of statistical indicators (mean, stddev etc.).

### 1.1. Detecting outliers
Indicators:

Z-score: Measures how far an observation deviates from the mean. It assumes that the data follows a normal distribution.

$$z_i = \frac{x_i - \overline{x}}{\sigma}$$

where $\sigma$ is the standard deviation. Since it relies on the mean and stddev, it is quite susceptible to outliers. Therefore, we use the modified Z-score:

$$mz_i = 0.6745 \cdot\frac{x_i-\tilde{x}}{\text{MAD}}$$

Where $\tilde{x}$ is the median and MAD the median absolute deviation value, sometimes also called the median absolute deviation from the median (MADFM); that is, the median ob the absolute deviations from the data's median: $\text{MAD} = \text{median}\vert x_i - \tilde{x} \vert$. An observation is considered a potential outlier if it falls more than 3.5 MAD from the median, i.e. $\vert mz_i \vert > 3.5$ The 3.5 is a recommended value, but I might just use 2.5. In turn

Sources:
Outlier detection: \href{https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm}{NIST}.
