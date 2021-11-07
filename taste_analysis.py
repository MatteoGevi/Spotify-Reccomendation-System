# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

# +
'''Chi Square Test'''

# Significance
significance = 0.01
# P-value
p = 1 - significance
# Degree of Freedom
dof = chi2_contingency(playlist)[2]
# Critical Value as Percet oint Function applied, which is the contrary of the Cumulative Distribution Function
critical_value = chi2.ppf(p, dof)

# Alternitevely, we can calculate a second p-value in case we need to corrobarate our test with another probability
p2 = chi2.cdf(critical_value, dof)
# -

# If the calculated Chi-square is greater than the critical value we reject the null hypothesis.
chi, pval, dof, exp = chi2_contingency(subjects)
print('p-value is: ', pval)
significance = 0.05
p = 1 - significance
critic_value = chi2.ppf(p, dof)
print('chi=%.6f, critical value=%.6f\n' % (chi, critic_value))if chi > critic_value:
    print("""At %.2f level of significance, we reject the null hypotheses and accept H1. They are not independent.""" % (significance))
else:
    print("""At %.2f level of significance, we accept the null hypotheses. They are independent.""" % (significance))


