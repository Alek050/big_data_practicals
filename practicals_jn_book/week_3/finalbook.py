import os
import platform

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler




file = "breast_cancer.parquet"
file_path  = os.path.join(os.path.dirname(__file__), file)
df_cancer = pd.read_parquet(file_path)


df_cancer.columns = df_cancer.columns.str.replace(" ", "_")

df_cancer["cclass"] = df_cancer["cclass"].astype('category')



mean_cols = [col for col in df_cancer.columns if 'mean' in col]
df_cancer[mean_cols] = df_cancer[mean_cols].fillna(df_cancer[mean_cols].mean())

worst_cols = [col for col in df_cancer.columns if 'worst' in col]
df_cancer[worst_cols] = df_cancer[worst_cols].ffill() # df.fillna(method='ffill') is depreciated

error_cols = [col for col in df_cancer.columns if 'error' in col]
df_cancer[error_cols] = df_cancer[error_cols].interpolate(method='linear')


mean_cols.append("cclass")  # we'll also need this column


df_subcancer = df_cancer[mean_cols].copy()


df_subbenign = df_subcancer[df_subcancer.cclass == "benign"]
df_submalig = df_subcancer[df_subcancer.cclass == "malignant"]


floatcols = df_subcancer.columns[:-1]
scaler = StandardScaler()
df_subcancer[floatcols] = scaler.fit_transform(df_subcancer[floatcols])


from numpy.polynomial.polynomial import polyfit

c = polyfit(df_subcancer.mean_concavity, df_subcancer.mean_concave_points, 1)
y_pred = c[0] + df_subcancer.mean_concavity * c[1]

corr = df_subcancer.corr(method='pearson', numeric_only=True)
scaler = StandardScaler()
X_std = scaler.fit_transform(df_cancer[df_cancer.columns[:-1]])

# In[26]:


from sklearn.decomposition import PCA
pca = PCA().fit(X_std)
# plt.figure(figsize=(10, 10))
# plt.plot(pca.explained_variance_ratio_, 'o-')
# plt.xlabel('number of components')
# plt.ylabel('explained variance ratio')
# plt.show()


pca = PCA(n_components=5)
X_PCA = pca.fit_transform(X_std)
print(pca.explained_variance_ratio_)
X_PCA.shape

exit()
# From the above output you can observe that the principal component 1 holds 44.2% of the information, the second holds only 19% of the information. the third, fourth and fifth contain 9.3%, 6.6% and 5.5%. Also, the other point to note is that while projecting thirty-dimensional data to a five-dimensional data, 15.4% information was lost. The last step is to see which original variables contribute to the components.
# 
# - **Inspect the components_ attribute of the pca, what does it contain?**
# ````{margin}
# ```{admonition} Tip
# :class: tip
# Notice that the weights have the same length as the amount of columns. The first weight belongs to the first column, second to the second etc. [Merge](https://pandas.pydata.org/docs/user_guide/merging.html) these DataFrames and print the top3!
# ```
# ````
# - **For now, we will only look into the first feature, make a DataFrame of the first feature**
# - **Make also a DataFrame of the columns names (all 30)**
# - **Find the three variables with the highest contribution to the first feature**

# In[28]:


print(pca.components_, '\n')

feature1 = pd.DataFrame(pca.components_[0], columns=['weight'])  # only first feature
columns = pd.DataFrame(df_cancer.columns[:-1], columns=['var'])  # take columns
merged = pd.concat([feature1, columns], axis=1)  # weights belong to columns
top3 = merged.sort_values(by='weight', ascending=False).head(3)  # top 3
print(top3)


# Good job, your CPU will thank you later! You can get a lot of information of your features. After the fancy regression analyses and you nicely performed PCA we will now move over to Hypothesis testing
# 
# ```{note}
# Remember, normally, Hypothesis testing comes before regression and PCA..
# ```
# 
# ## Hypothesis testing
# 
# Point estimates such as the mean and median are a nice way to describe the population, but the difference could be cause only by chance, because of the variability of both estimates. R.A. Fisher (1890–1962) proposed an alternative, known as hypothesis testing, that is based on the concept of statistical significance. We can see that the mean_radius for benign and malignant cells is different, but is this due to chance or can we consider them as belonging to two different populations. Then, the relevant question is: Are the observed effects real or not?
# 
# Technically, the question is usually translated to: Were the observed effects statistically significant?
# The process of determining the statistical significance of an effect is called hypothesis testing.
# 
# This process starts by simplifying the options into two competing hypotheses:
# 
# >- H0: the effect we have observed is due to chance (due to the specific sample bias). 
# >
# >- H1: the effect we have observed is due to real differences between the groups.
# 
# We will not discard H0 unless the observed effect is implausible under H0.
# 
# Think of an example what the H0 and H1 are in our dataset.
# 
# ### Assignment 12: confidence interval
# 
# We can use the concept represented by confidence intervals to measure the plausibility of a hypothesis
# 
# If the interval spreads out 1.96 standard errors from a normally distributed point
# estimate, intuitively we can say that we are roughly 95% confident that we have
# captured the true parameter. (we will make some assumptions here which we will address later)
# 
# CI = [mean − 1.96 × $SE$, mean + 1.96 × $SE$]
# 
# ```{note}
# - If the P value is less than your significance (alpha) level, the hypothesis test is statistically significant.
# - If the confidence interval does not contain the null hypothesis value, the results are statistically significant.
# - If the P value is less than alpha, the confidence interval will not contain the null hypothesis value.
# ```
# 
# ````{margin}
# ```{admonition} Tip
# :class: tip
# The $SE$ can be calculated with se = sample.std() / np.sqrt(len(sample))
# ```
# ````
# - **Calculate the confidence interval for the malignant cells.**
# - **Print the mean mean_radius for the malignant cells and the benign cells.**
# - **Are the benign cells positioned in the confidence interval of the malignant cells?**
# 
# You should get something like this:

# In[29]:


n = len(df_submalig.mean_radius)
mean = df_submalig.mean_radius.mean()
s = df_submalig.mean_radius.std()
ci = [mean - s*1.96/np.sqrt(n), mean + s*1.96/np.sqrt(n)]

print("Mean radius of malignant cells:", df_submalig.mean_radius.mean().round(2))
print("Mean radius of benign cells:", df_subbenign.mean_radius.mean().round(2))
print("Confidence interval for malignant cells", [x_value.round(2) for x_value in ci])


# ### Assignment 13: using statistical testing
# 
# Statistical testing in Python is relatively easy. There are two main packages available for regular frequentist statistical testing: [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) and [statsmodels](https://www.statsmodels.org/stable/index.html). If you like R you will like statsmodels. We will however use scipy.stats as it is the most straightforward.
# 
# First, a simple independent t-test. The test measures whether the average (expected) value differs significantly across samples. If we observe a large p-value, for example larger than 0.05 or 0.1, then we cannot reject the null hypothesis (H0) of identical average scores. If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages.
# 
# Let's start by comparing mean_radius between malignant and benign cases. To not anger the statisticians too much we should probably test for a normal distribution and equality of variances first. 
# 
# - **Import shapiro, levene, normaltest, skewtest, and kurtosistest from the scipy.stats sublibrary. Alternatively, just import the entire library with ``from scipy.stats import *``. Be careful with the * strategy, but in this case it is a reasonable thing to do.**
# 
# - **Check whether the data is normally distributed by performing all three tests on both the benign malignent data.**
# 
# You should get something like this:

# In[30]:


for df in [df_submalig, df_subbenign]:
    print("Shapiro-Wilks result:", shapiro(df["mean_radius"]))
    print(normaltest(df["mean_radius"]))
    print(skewtest(df["mean_radius"]))
    print(kurtosistest(df["mean_radius"]), "\n")


# OK, we have some issues with normality here, so we will use a non-parametric test later. First, let's do the parametric one anyways.
# 
# - **Appease statisticians by performing a levenes test, simply enter the date into ``levene()``.**
# ````{margin}
# ```{admonition} Tip
# :class: tip
# Make use of an if statement.
# ```python
# if p < 0.05:
#     ...
# else:
#     ...
# ```
# ````
# - **Make a variable that is either True or False based on the result of Levene's test.**
# 
# - **Perform a ``.ttest_ind()`` and specify the equal_vars parameter with your just made True or False variable.**
# 
# - **Print something based on the result of your test.**
# 
# You should get something like this:

# In[31]:


l_test = levene(df_subbenign["mean_radius"], df_submalig["mean_radius"])
print(l_test)

p = l_test[1]
if p < 0.05:
    print("Equal variances not assumed")
    e_vars = False
else:
    print("Equal variances assumed")
    e_vars = True

ttest_result = ttest_ind(df_subbenign["mean_radius"], df_submalig["mean_radius"], equal_var=e_vars)
print(ttest_result)

p = ttest_result[1]
if p < 0.05:
    print("The difference is statistically significant!")
else:
    print("The difference is not statistically significant")


# This is pretty much what we expected from our figures and our simulation. Also, with this amount of data it is pretty hard not to find a significantly different result. Now do the same thing with a non-parametric test. Note that the data was not normally distributed, so we should have actually tested with a Mann-Whitney U in the first place.
# 
# - **Repeat the test with a Mann-Whitney U from the scipy library. Print something based on the p-value, your alpha is 0.05, as is tradition.**
# 
# You should get something like this:

# In[32]:


mann_result = mannwhitneyu(df_subbenign["mean_radius"], df_submalig["mean_radius"])  # non-parametric test
print(mann_result)

p = mann_result[1]
if p < 0.05:
    print("The difference is statistically significant😎")
else:
    print("The difference is not statistically significant")


# Another way to go about this is to use a linear model. The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean. The test is applied to samples from two or more groups, possibly with differing sizes. We can also apply the ANOVA on this data. The only difference is that you will get an F-statistic instead of a T-statistic.
# 
# The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
# ```{note}
# The samples are independent. Each sample is from a normally distributed population. The population standard deviations of the groups are all equal. This property is known as homoscedasticity.
# ```
# 
# - **Perform the same test using ``f_oneway`` from the scipy.stats sublibrary. Print something based on the p-value.**
# 
# - **The assumption of normality is violated. ANOVA is very robust IF we have the same sample size for both groups. This is not the case, therefore you should repeat this exercise with the non-parametric variant. Use the ``kruskal`` test from the same library.**
# 
# You should get something like this:

# In[33]:


anova_result = f_oneway(df_subbenign["mean_radius"], df_submalig["mean_radius"])
print(anova_result)

p = anova_result[1]
if p < 0.05:
    print("The difference is statistically significant 😍")
else:
    print("The difference is not statistically significant")

kruskal_result = kruskal(df_subbenign["mean_radius"], df_submalig["mean_radius"])
print(kruskal_result)

p = kruskal_result[1]
if p < 0.05:
    print("The difference is statistically significant 😍")
else:
    print("The difference is not statistically significant")


# Good Job! Now you are ready for the real machine learning fun, see you next week!
