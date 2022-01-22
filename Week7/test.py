from scipy.stats import ttest_1samp
import numpy as np

ages = np.genfromtxt("ages.csv")

print(ages)

ages_mean = np.mean(ages)

print(ages_mean)

tset, pval = ttest_1samp(ages, 30)
print("p-values", pval)
if pval < 0.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")