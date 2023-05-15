import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2
import matplotlib.pyplot as plt



baseball =pd.read_csv('baseball.csv')



# View summary statistics of the numeric columns
baseball.describe()



baseball.head()


baseball.tail()


baseball.info()


baseball.shape


# ### Dataset was checked for missing data
baseball.isnull().sum



baseball_num=baseball.select_dtypes(include='number').columns



baseball_num


# ### Missing data was filled by imputeed using the mean of each column
baseball[baseball_num]=baseball[baseball_num].fillna(baseball[baseball_num].mean())


baseball


baseball.describe()


# ## Data Visualization

# Plot number of wins vs. number of runs scored
plt.scatter(baseball["RS"], baseball["W"])
plt.xlabel("Runs Scored")
plt.ylabel("Number of Wins")
plt.show()


# Plot histogram of number of wins
plt.hist(baseball["W"], bins=20)
plt.xlabel("Number of Wins")
plt.ylabel("Frequency")
plt.show()


# corr_matrix=baseball.corr()
# corr_matrix

# In[16]:


# Create a new column for decade
baseball["Decade"] = (baseball["Year"] // 10) * 10



# Create a contingency table of number of wins by decade
wins_by_decade = pd.crosstab(baseball["Decade"], baseball["W"])
wins_by_decade

## Chi-Square Goodness-of-Fit test on baseball dataset


# ### Null hypothsis(H0):There is no significant difference in the number of wins by decade.
# ### Alternative hypothesis(Ha):There is a significant difference in the number of wins by decade.

# In[18]:


# Calculate expected values
expected_wins = wins_by_decade.sum().sum() / wins_by_decade.size
expected_values = pd.DataFrame(expected_wins, index=wins_by_decade.index, columns=wins_by_decade.columns)


# In[19]:


# Perform Chi-Square Goodness-of-Fit test
from scipy.stats import chisquare
chi2, p_value = chisquare(f_obs=wins_by_decade, f_exp=expected_values, axis=None)
print("Chi-Square test statistic:", chi2)
print("p-value:", p_value)


# In[20]:


if p_value < 0.05:
    print('Reject the null hypothesis: There is no significant difference in the number of wins by decade.')
else:
    print('Fail to reject the null hypothesis: There is a significant difference in the number of wins by decade.')


# ## Anova Two Way Test on crop datset

# ### Null hypothesis for fertilizer: The mean yield is the same for all levels of fertilizer.
# ### Alternative hypothesis for fertilizer: The mean yield is different for at least one level of fertilizer.
# ### Null hypothesis for density: The mean yield is the same for all levels of density.
# ### Alternative hypothesis for density: The mean yield is different for at least one level of density.
# ### Null hypothesis for interaction: The effect of fertilizer on yield is the same for all levels of density, and the effect of density on yield is the same for all levels of fertilizer.
# ### Alternative hypothesis for interaction: The effect of fertilizer on yield is different for at least one level of density, or the effect of density on yield is different for at least one level of fertilizer.

# In[21]:


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols



crop=pd.read_csv('crop_data.csv')


crop


crop.describe()


# Convert variables to factors
crop["density"] = crop["density"].astype("category")
crop["fertilizer"] = crop["fertilizer"].astype("category")
crop["block"] = crop["block"].astype("category")


crop = crop.rename(columns={"yield": "yieeld"})


# Fit the two-way ANOVA model
model = ols('yieeld ~ C(density) + C(fertilizer) + C(density):C(fertilizer)', data=crop).fit()
anova_table = sm.stats.anova_lm(model, typ=2)


# Print the ANOVA table
print(anova_table)
# Extract the F-statistic and p-value for each effect
f_density, p_density = anova_table.loc['C(density)', ['F', 'PR(>F)']]
f_fertilizer, p_fertilizer = anova_table.loc['C(fertilizer)', ['F', 'PR(>F)']]
f_interaction, p_interaction = anova_table.loc['C(density):C(fertilizer)', ['F', 'PR(>F)']]

# Print the F-statistics and p-values for each effect
print(f"Density: F = {f_density}, p = {p_density}")
print(f"Fertilizer: F = {f_fertilizer}, p = {p_fertilizer}")
print(f"Interaction: F = {f_interaction}, p = {p_interaction}")


# Define the significance level
alpha = 0.05

# Test the interaction effect
if p_interaction < alpha:
    print("Reject the null hypothesis that the effect of fertilizer on yield is the same for all levels of density, and the effect of density on yield is the same for all levels of fertilizer.")
else:
    print("Fail to reject the null hypothesis that the effect of fertilizer on yield is the same for all levels of density, and the effect of density on yield is the same for all levels of fertilizer.")
# Test the effect of gas type
if p_density < alpha:
    print("Reject the null hypothesis that the mean yield is the same for all levels of density.")
else:
    print("Fail to reject the null hypothesis that the mean yield is the same for all levels of density.")

# Test the effect of drive type
if p_fertilizer < alpha:
    print("Reject the null hypothesis that the mean yield is the same for all levels of fertilizer.")
else:
    print("Fail to reject the null hypothesis that the mean yield is the same for all levels of fertilizer.")

