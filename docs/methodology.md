# Credit Risk Modeling Engine — Mathematical Methodology

## 1. Purpose and scope

The Credit Risk Modeling Engine provides a transparent and reproducible framework for measuring, simulating, and interpreting credit risk.

The engine combines:

- probability of default estimation;
- loss given default and exposure at default;
- expected and unexpected credit loss;
- logistic credit scoring;
- credit-rating transition matrices;
- multi-period default probabilities;
- correlated-default Monte Carlo simulation;
- Credit Value at Risk;
- Expected Shortfall;
- economic capital;
- portfolio concentration analysis;
- stress testing;
- graphical risk diagnostics.

The models are designed for educational, analytical, and portfolio-research purposes. They are not a substitute for regulatory models, audited financial systems, or professional credit decisions.

---

## 2. Mathematical notation

Consider a credit portfolio containing \(n\) borrowers or credit exposures.

For borrower \(i\):

| Symbol | Meaning |
|---|---|
| \(PD_i\) | Probability of default |
| \(LGD_i\) | Loss given default |
| \(EAD_i\) | Exposure at default |
| \(RR_i\) | Recovery rate |
| \(EL_i\) | Expected loss |
| \(UL_i\) | Unexpected loss |
| \(D_i\) | Default indicator |
| \(L_i\) | Credit loss |
| \(w_i\) | Portfolio exposure weight |
| \(\rho_i\) | Asset correlation |
| \(R_i\) | Credit rating |
| \(S_i\) | Economic sector |

The exposure weight is:

\[
w_i =
\frac{EAD_i}
{\sum_{j=1}^{n} EAD_j}
\]

and therefore:

\[
\sum_{i=1}^{n} w_i = 1
\]

All probabilities and loss rates are represented as decimal values between zero and one.

---

## 3. Fundamental credit-risk components

### 3.1 Probability of Default

The probability of default measures the likelihood that a borrower will fail to meet its contractual obligations during a specified time horizon.

\[
PD_i = P(D_i = 1)
\]

where:

\[
D_i =
\begin{cases}
1, & \text{if borrower } i \text{ defaults} \\
0, & \text{otherwise}
\end{cases}
\]

For example, if:

\[
PD_i = 0.03
\]

the borrower has an estimated 3% probability of default during the model horizon.

The complementary probability is the probability of survival:

\[
P(D_i = 0) = 1-PD_i
\]

The model assumes that each probability of default satisfies:

\[
0 \leq PD_i \leq 1
\]

---

### 3.2 Loss Given Default

Loss Given Default represents the proportion of the exposure that is lost if default occurs.

\[
LGD_i = 1-RR_i
\]

where \(RR_i\) is the recovery rate.

For example, if the lender expects to recover 40% of an exposure:

\[
RR_i = 0.40
\]

then:

\[
LGD_i = 1-0.40 = 0.60
\]

Therefore, 60% of the outstanding exposure is expected to be lost if the borrower defaults.

The model requires:

\[
0 \leq LGD_i \leq 1
\]

---

### 3.3 Exposure at Default

Exposure at Default represents the amount economically exposed when default occurs.

For a fully drawn loan:

\[
EAD_i = \text{Outstanding balance}_i
\]

For a revolving facility, a more general representation is:

\[
EAD_i =
\text{Drawn amount}_i
+
CCF_i \times
\text{Undrawn commitment}_i
\]

where \(CCF_i\) is the credit conversion factor.

The engine requires:

\[
EAD_i \geq 0
\]

---

## 4. Expected credit loss

### 4.1 Individual expected loss

The expected loss of borrower \(i\) is:

\[
EL_i = PD_i \times LGD_i \times EAD_i
\]

This formula combines:

1. the probability that default occurs;
2. the percentage lost after default;
3. the amount exposed at default.

Example:

\[
PD_i = 0.03
\]

\[
LGD_i = 0.45
\]

\[
EAD_i = \$1{,}000{,}000
\]

Then:

\[
EL_i =
0.03 \times 0.45 \times 1{,}000{,}000
=
\$13{,}500
\]

The result is a probability-weighted average loss, not a prediction that the exact loss will necessarily occur.

---

### 4.2 Portfolio expected loss

For a portfolio containing \(n\) exposures:

\[
EL_{\text{portfolio}}
=
\sum_{i=1}^{n}
PD_i LGD_i EAD_i
\]

The expected-loss rate is:

\[
EL_{\text{rate}}
=
\frac{EL_{\text{portfolio}}}
{\sum_{i=1}^{n}EAD_i}
\]

Expected loss is normally interpreted as the average credit cost that should be covered through:

- interest-rate spreads;
- credit provisions;
- loan-loss reserves;
- pricing policies.

---

## 5. Unexpected loss

Expected loss measures the average loss. Unexpected loss measures uncertainty around that average.

For an individual Bernoulli default exposure with fixed \(LGD\) and \(EAD\):

\[
L_i = D_i \times LGD_i \times EAD_i
\]

Since \(D_i\) follows a Bernoulli distribution:

\[
E[D_i] = PD_i
\]

and:

\[
\operatorname{Var}(D_i)
=
PD_i(1-PD_i)
\]

Therefore:

\[
\operatorname{Var}(L_i)
=
(LGD_i EAD_i)^2
PD_i(1-PD_i)
\]

and individual unexpected loss is:

\[
UL_i =
LGD_i EAD_i
\sqrt{PD_i(1-PD_i)}
\]

Under the simplifying assumption of independent defaults:

\[
UL_{\text{portfolio}}
=
\sqrt{
\sum_{i=1}^{n}
UL_i^2
}
\]

However, real defaults are not independent. Borrowers can be affected simultaneously by recessions, interest rates, unemployment, commodity prices, and sector-specific shocks.

For correlated exposures:

\[
\operatorname{Var}(L_{\text{portfolio}})
=
\sum_{i=1}^{n}\operatorname{Var}(L_i)
+
2\sum_{i<j}
\operatorname{Cov}(L_i,L_j)
\]

The correlated-default simulation included in the engine captures this dependence more realistically.

---

## 6. Logistic credit-scoring model

### 6.1 Objective

The credit-scoring model estimates a borrower’s probability of default using financial and behavioral characteristics.

Let the borrower feature vector be:

\[
\mathbf{x}_i =
(x_{i1},x_{i2},\ldots,x_{ik})
\]

Possible explanatory variables include:

- income;
- debt-to-income ratio;
- credit utilization;
- delinquency history;
- loan-to-value ratio;
- account age;
- payment behavior;
- number of previous defaults.

---

### 6.2 Linear predictor

Logistic regression first creates a linear score:

\[
z_i =
\beta_0
+
\beta_1x_{i1}
+
\beta_2x_{i2}
+\cdots+
\beta_kx_{ik}
\]

where:

- \(\beta_0\) is the intercept;
- \(\beta_j\) is the coefficient associated with feature \(j\).

---

### 6.3 Logistic transformation

The linear score is converted into a probability using the logistic function:

\[
PD_i =
\sigma(z_i)
=
\frac{1}{1+e^{-z_i}}
\]

This guarantees:

\[
0 < PD_i < 1
\]

The log-odds representation is:

\[
\log
\left(
\frac{PD_i}{1-PD_i}
\right)
=
\beta_0
+
\sum_{j=1}^{k}\beta_jx_{ij}
\]

A positive coefficient increases the borrower’s log-odds of default, while a negative coefficient decreases them, holding the other variables constant.

---

### 6.4 Maximum-likelihood estimation

For observed outcomes \(y_i \in \{0,1\}\), the likelihood function is:

\[
\mathcal{L}(\boldsymbol{\beta})
=
\prod_{i=1}^{n}
PD_i^{y_i}
(1-PD_i)^{1-y_i}
\]

The log-likelihood is:

\[
\ell(\boldsymbol{\beta})
=
\sum_{i=1}^{n}
\left[
y_i\log(PD_i)
+
(1-y_i)\log(1-PD_i)
\right]
\]

The coefficients are estimated by maximizing this log-likelihood or, equivalently, minimizing the negative log-likelihood.

---

### 6.5 Classification rule

A probability threshold \(c\) converts the estimated probability into a class:

\[
\widehat{y}_i =
\begin{cases}
1, & \widehat{PD}_i \geq c \\
0, & \widehat{PD}_i < c
\end{cases}
\]

A typical initial threshold is:

\[
c=0.50
\]

However, credit-risk applications often use lower thresholds because failing to identify a default can be substantially more expensive than investigating a false alert.

---

### 6.6 Evaluation metrics

The confusion matrix contains:

- true positives (\(TP\));
- true negatives (\(TN\));
- false positives (\(FP\));
- false negatives (\(FN\)).

Accuracy:

\[
\text{Accuracy}
=
\frac{TP+TN}
{TP+TN+FP+FN}
\]

Precision:

\[
\text{Precision}
=
\frac{TP}
{TP+FP}
\]

Recall:

\[
\text{Recall}
=
\frac{TP}
{TP+FN}
\]

Specificity:

\[
\text{Specificity}
=
\frac{TN}
{TN+FP}
\]

F1 score:

\[
F_1 =
2
\frac{
\text{Precision}\times\text{Recall}
}{
\text{Precision}+\text{Recall}
}
\]

ROC-AUC measures the model’s ability to rank defaulters above non-defaulters across all possible classification thresholds.

An AUC near 0.50 indicates limited discriminatory power, while a value closer to 1.00 indicates stronger ranking performance.

---

## 7. Credit-rating transition model

### 7.1 Transition matrix

A credit-rating transition matrix describes the probability that an obligor moves from one rating category to another over a fixed horizon.

Let the rating states be:

\[
\mathcal{R}
=
\{AAA,AA,A,BBB,BB,B,CCC,D\}
\]

where \(D\) represents default.

The one-period transition matrix is:

\[
P =
\begin{bmatrix}
p_{11} & p_{12} & \cdots & p_{1m} \\
p_{21} & p_{22} & \cdots & p_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
p_{m1} & p_{m2} & \cdots & p_{mm}
\end{bmatrix}
\]

Each element is:

\[
p_{ij}
=
P(R_{t+1}=j \mid R_t=i)
\]

Every row must satisfy:

\[
p_{ij}\geq 0
\]

and:

\[
\sum_{j=1}^{m}p_{ij}=1
\]

---

### 7.2 Default as an absorbing state

Default is generally modeled as an absorbing state:

\[
p_{DD}=1
\]

and:

\[
p_{Dj}=0
\quad \text{for } j\neq D
\]

Once an obligor enters the default state, it remains there within the transition model.

---

### 7.3 Multi-period transition probabilities

Assuming a time-homogeneous Markov process, the \(h\)-period transition matrix is:

\[
P^{(h)} = P^h
\]

For example, the five-year transition matrix is:

\[
P^{(5)} = P^5
\]

The cumulative probability of default over \(h\) periods for an initial rating \(r\) is:

\[
PD_r^{(h)}
=
\left[P^h\right]_{r,D}
\]

This probability includes all possible migration paths leading to default.

---

### 7.4 Markov assumption

The transition model assumes:

\[
P(R_{t+1}\mid R_t,R_{t-1},\ldots)
=
P(R_{t+1}\mid R_t)
\]

Therefore, the next rating depends only on the current rating and not on the complete previous rating history.

This is useful computationally but can be restrictive because actual credit migrations may depend on:

- the direction of previous migrations;
- time spent in a rating;
- macroeconomic conditions;
- sector behavior;
- rating momentum.

---

### 7.5 Rating simulation

For a borrower currently in rating \(r\), the next rating is sampled from the categorical distribution defined by row \(r\):

\[
R_{t+1}
\sim
\operatorname{Categorical}
(p_{r1},p_{r2},\ldots,p_{rm})
\]

Repeating this procedure produces simulated credit-rating paths and possible default events.

A fixed random seed makes simulation results reproducible.

---

## 8. Correlated-default Monte Carlo model

### 8.1 Motivation

Independent-default models underestimate portfolio tail risk when borrowers are exposed to common economic conditions.

The engine uses a one-factor Gaussian latent-variable model to represent default dependence.

---

### 8.2 Systematic and idiosyncratic risk

For borrower \(i\), define latent credit quality:

\[
A_i =
\sqrt{\rho_i}\,Z
+
\sqrt{1-\rho_i}\,\varepsilon_i
\]

where:

\[
Z \sim N(0,1)
\]

is the common systematic factor and:

\[
\varepsilon_i \sim N(0,1)
\]

is the borrower-specific factor.

The variables \(Z\) and \(\varepsilon_i\) are mutually independent.

The asset-correlation parameter satisfies:

\[
0\leq \rho_i < 1
\]

---

### 8.3 Default threshold

The borrower defaults when latent credit quality falls below a threshold:

\[
A_i <
\Phi^{-1}(PD_i)
\]

Therefore:

\[
D_i =
\mathbf{1}
\left\{
A_i <
\Phi^{-1}(PD_i)
\right\}
\]

where:

- \(\Phi\) is the standard normal cumulative distribution;
- \(\Phi^{-1}\) is its inverse;
- \(\mathbf{1}\{\cdot\}\) is an indicator function.

Because \(A_i\sim N(0,1)\):

\[
P(D_i=1)
=
P\left(
A_i<\Phi^{-1}(PD_i)
\right)
=
PD_i
\]

Thus, the simulation preserves each borrower’s unconditional probability of default.

---

### 8.4 Interpretation of correlation

When \(\rho_i=0\):

\[
A_i=\varepsilon_i
\]

and defaults are driven entirely by borrower-specific shocks.

As \(\rho_i\) increases, the systematic factor becomes more important. Borrowers are then more likely to default together during unfavorable economic scenarios.

Higher correlation usually produces:

- greater loss clustering;
- heavier portfolio-loss tails;
- higher Credit VaR;
- higher Expected Shortfall;
- greater economic capital.

---

### 8.5 Scenario loss

For simulation \(s\), portfolio credit loss is:

\[
L^{(s)}
=
\sum_{i=1}^{n}
D_i^{(s)}
LGD_i
EAD_i
\]

The Monte Carlo loss distribution is:

\[
\left\{
L^{(1)},L^{(2)},\ldots,L^{(M)}
\right\}
\]

where \(M\) is the number of simulations.

The simulated average loss is:

\[
\widehat{EL}
=
\frac{1}{M}
\sum_{s=1}^{M}L^{(s)}
\]

As the number of simulations increases, the estimated average should approach analytical expected loss:

\[
EL_{\text{analytical}}
=
\sum_{i=1}^{n}
PD_iLGD_iEAD_i
\]

---

### 8.6 Monte Carlo algorithm

For every simulation:

1. Generate one common systematic factor \(Z\).
2. Generate one idiosyncratic shock \(\varepsilon_i\) for every borrower.
3. Calculate latent credit quality \(A_i\).
4. Compare \(A_i\) with the default threshold.
5. Determine each borrower’s default indicator.
6. Calculate losses using \(LGD_iEAD_i\).
7. Aggregate losses across the portfolio.
8. Store the total simulated loss.

After all simulations, calculate the required distributional risk measures.

---

## 9. Credit Value at Risk

At confidence level \(\alpha\), Credit VaR is the \(\alpha\)-quantile of the loss distribution:

\[
VaR_{\alpha}(L)
=
\inf
\left\{
\ell:
P(L\leq \ell)\geq\alpha
\right\}
\]

For example, 99% Credit VaR is a loss threshold exceeded in approximately 1% of simulated scenarios.

If:

\[
VaR_{0.99}=\$1{,}147{,}500
\]

the interpretation is that approximately 99% of modeled losses are no greater than \$1,147,500 over the selected horizon.

VaR identifies the beginning of the extreme-loss region but does not measure the average severity beyond that threshold.

---

## 10. Expected Shortfall

Expected Shortfall measures the average loss in the tail beyond VaR:

\[
ES_{\alpha}
=
E
\left[
L
\mid
L\geq VaR_{\alpha}(L)
\right]
\]

Its empirical estimator is:

\[
\widehat{ES}_{\alpha}
=
\frac{
\sum_{s=1}^{M}
L^{(s)}
\mathbf{1}
\left\{
L^{(s)}
\geq
\widehat{VaR}_{\alpha}
\right\}
}{
\sum_{s=1}^{M}
\mathbf{1}
\left\{
L^{(s)}
\geq
\widehat{VaR}_{\alpha}
\right\}
}
\]

Expected Shortfall provides information about tail severity that VaR alone does not describe.

Generally:

\[
ES_{\alpha}
\geq
VaR_{\alpha}
\]

---

## 11. Economic capital

Economic capital represents the capital buffer required to absorb unexpected loss at the selected confidence level.

The engine defines economic capital as:

\[
EC_{\alpha}
=
VaR_{\alpha}
-
EL
\]

where \(EL\) may be the analytical or simulated expected loss.

Conceptually:

- expected loss should be covered by pricing and provisions;
- unexpected loss should be supported by capital.

If:

\[
VaR_{0.99}
=
\$1{,}147{,}500
\]

and:

\[
EL
=
\$124{,}700
\]

then:

\[
EC_{0.99}
=
1{,}147{,}500
-
124{,}700
=
\$1{,}022{,}800
\]

---

## 12. Portfolio concentration risk

### 12.1 Borrower concentration

A borrower’s exposure share is:

\[
w_i =
\frac{EAD_i}
{\sum_{j=1}^{n}EAD_j}
\]

The largest-borrower share is:

\[
w_{\max}
=
\max_i(w_i)
\]

A large value indicates dependence on a single borrower.

---

### 12.2 Herfindahl–Hirschman Index

The Herfindahl–Hirschman Index measures concentration:

\[
HHI =
\sum_{i=1}^{n}w_i^2
\]

The HHI satisfies:

\[
\frac{1}{n}
\leq HHI
\leq 1
\]

Interpretation:

- values closer to \(1/n\) indicate greater diversification;
- values closer to 1 indicate greater concentration;
- \(HHI=1\) means the entire portfolio belongs to one exposure.

The effective number of equally sized borrowers can be approximated by:

\[
N_{\text{effective}}
=
\frac{1}{HHI}
\]

---

### 12.3 Sector concentration

For sector \(s\), total exposure is:

\[
EAD_s =
\sum_{i:S_i=s}EAD_i
\]

The sector exposure share is:

\[
w_s =
\frac{EAD_s}
{\sum_{i=1}^{n}EAD_i}
\]

Sector expected loss is:

\[
EL_s =
\sum_{i:S_i=s}
PD_iLGD_iEAD_i
\]

A portfolio may contain many borrowers but still be poorly diversified if they belong to the same economic sector.

---

### 12.4 Risk contribution

Borrower \(i\)'s contribution to expected loss is:

\[
RC_i^{EL}
=
PD_iLGD_iEAD_i
\]

Its proportional contribution is:

\[
\widetilde{RC}_i^{EL}
=
\frac{RC_i^{EL}}
{\sum_{j=1}^{n}RC_j^{EL}}
\]

This identifies the borrowers responsible for the largest proportion of expected credit cost.

---

## 13. Stress testing

### 13.1 Purpose

Stress testing evaluates portfolio behavior under adverse but plausible conditions.

Unlike historical estimation, stress testing asks:

> What could happen if economic and credit conditions deteriorate?

---

### 13.2 Probability-of-default stress

A multiplicative PD stress is:

\[
PD_i^{\text{stress}}
=
\min
\left(
1,
m_{PD}PD_i
\right)
\]

where \(m_{PD}\geq1\) is the PD multiplier.

The upper bound prevents stressed probabilities from exceeding 100%.

---

### 13.3 Loss-given-default stress

LGD can be stressed using either a multiplier:

\[
LGD_i^{\text{stress}}
=
\min
\left(
1,
m_{LGD}LGD_i
\right)
\]

or an additive shock:

\[
LGD_i^{\text{stress}}
=
\min
\left(
1,
LGD_i+\Delta LGD
\right)
\]

This reflects the possibility that collateral values and recovery rates deteriorate during recessions.

---

### 13.4 Exposure-at-default stress

Exposure may rise during periods of financial pressure because borrowers draw unused facilities:

\[
EAD_i^{\text{stress}}
=
m_{EAD}EAD_i
\]

where \(m_{EAD}\geq1\).

---

### 13.5 Stressed expected loss

Under a combined scenario:

\[
EL_{\text{stress}}
=
\sum_{i=1}^{n}
PD_i^{\text{stress}}
LGD_i^{\text{stress}}
EAD_i^{\text{stress}}
\]

The increase relative to the baseline is:

\[
\Delta EL =
EL_{\text{stress}}
-
EL_{\text{baseline}}
\]

The relative change is:

\[
\%\Delta EL
=
\frac{
EL_{\text{stress}}
-
EL_{\text{baseline}}
}{
EL_{\text{baseline}}
}
\times 100
\]

---

## 14. Correlation-sensitivity analysis

Correlation sensitivity evaluates the model under several asset-correlation assumptions:

\[
\rho
\in
\{0.00,0.10,0.20,0.35,0.50\}
\]

For each value, the engine recalculates:

- simulated expected loss;
- Credit VaR;
- Expected Shortfall;
- economic capital;
- loss-distribution dispersion.

Expected loss should remain relatively stable because marginal default probabilities do not change.

Tail measures generally increase with correlation because defaults become more synchronized.

This analysis demonstrates that diversification depends not only on the number of borrowers but also on the dependence between them.

---

## 15. Model validation

A credit-risk model should be evaluated across multiple dimensions.

### 15.1 Input validation

The engine verifies that:

\[
0\leq PD_i\leq1
\]

\[
0\leq LGD_i\leq1
\]

\[
EAD_i\geq0
\]

It also checks:

- compatible array lengths;
- finite numerical values;
- valid transition-matrix dimensions;
- row probabilities summing to one;
- admissible confidence levels;
- positive simulation counts;
- valid correlation parameters.

---

### 15.2 Analytical consistency

The simulated mean should be reasonably close to analytical expected loss:

\[
\widehat{EL}_{MC}
\approx
\sum_{i=1}^{n}
PD_iLGD_iEAD_i
\]

Large differences may indicate:

- insufficient simulations;
- incorrect default thresholds;
- inconsistent random-number generation;
- loss aggregation errors.

---

### 15.3 Reproducibility

A fixed random seed ensures:

\[
\text{same inputs}
+
\text{same seed}
\Rightarrow
\text{same simulated results}
\]

This is essential for:

- automated testing;
- model review;
- debugging;
- auditability;
- comparison of alternative assumptions.

---

### 15.4 Unit testing

The automated test suite verifies:

- formulas and known numerical examples;
- boundary conditions;
- invalid-input handling;
- data-class structures;
- transition-matrix rules;
- deterministic seeded simulations;
- portfolio aggregation;
- plotting functions;
- relationships between risk measures.

Examples of logical relationships tested include:

\[
ES_{\alpha}
\geq
VaR_{\alpha}
\]

and:

\[
VaR_{\alpha}
\geq
EL
\]

for ordinary non-degenerate credit portfolios.

---

## 16. Main assumptions

The methodology relies on the following assumptions:

1. PD, LGD, and EAD are known or estimated externally.
2. LGD and EAD are deterministic within each simulation unless explicitly stressed.
3. The simulation horizon is consistent across all exposures.
4. Default occurs once during the selected horizon.
5. The systematic factor follows a standard normal distribution.
6. Idiosyncratic shocks are normally distributed and mutually independent.
7. Dependence is represented by a one-factor Gaussian structure.
8. Transition probabilities are time-homogeneous.
9. Rating evolution satisfies the Markov property.
10. Default is an absorbing rating state.
11. Portfolio composition remains fixed during the analysis horizon.
12. Transaction costs, taxes, prepayments, and dynamic recoveries are excluded.

---

## 17. Model limitations

The engine is intentionally transparent and educational. Important limitations include:

- PD estimates may be affected by limited or biased historical data.
- Logistic regression assumes a linear relationship in log-odds.
- Class imbalance may distort classification metrics.
- Rating-transition probabilities may change across economic cycles.
- The Markov assumption ignores full rating history.
- Gaussian dependence may underestimate very extreme joint defaults.
- A single systematic factor cannot represent every sector and geographic shock.
- Fixed LGD does not capture recovery uncertainty.
- Fixed EAD does not fully represent credit-line drawdowns.
- Monte Carlo tail estimates contain sampling error.
- Stress scenarios are assumption-driven rather than forecasts.
- Concentration measures do not replace a complete dependence model.
- Regulatory capital calculations may require institution-specific rules and supervisory approval.

---

## 18. Possible extensions

Future versions could include:

- through-the-cycle and point-in-time PD models;
- macroeconomic satellite models;
- multiple systematic risk factors;
- sector-specific asset correlations;
- stochastic LGD and EAD;
- copula-based default dependence;
- survival and hazard-rate models;
- Bayesian credit models;
- calibration using real default data;
- IFRS 9 staging and lifetime expected credit loss;
- Basel regulatory-capital calculations;
- model backtesting and calibration diagnostics;
- explainability using SHAP values;
- web-based interactive dashboards;
- database integration and automated reporting.

---

## 19. Interpretation framework

The engine answers several complementary questions:

| Measure | Question answered |
|---|---|
| PD | How likely is default? |
| LGD | What percentage may be lost after default? |
| EAD | How much is exposed when default occurs? |
| Expected loss | What is the average credit cost? |
| Unexpected loss | How uncertain is the loss? |
| Credit VaR | Where does the extreme-loss region begin? |
| Expected Shortfall | How severe are losses inside that region? |
| Economic capital | How much capital supports unexpected loss? |
| HHI | How concentrated is the portfolio? |
| Transition matrix | How may ratings migrate over time? |
| Stress test | What happens under adverse assumptions? |

No single metric describes the complete risk profile. Sound analysis combines probability, severity, dependence, concentration, migration, and stress.

---

## 20. Educational disclaimer

This software is a quantitative educational simulator.

Its outputs depend on model assumptions, input quality, calibration choices, and random simulation. Results must not be interpreted as guaranteed forecasts, investment recommendations, lending decisions, accounting provisions, or regulatory-capital determinations.

Production use would require:

- independently validated data;
- formal model governance;
- calibration and backtesting;
- documented approvals;
- access controls;
- audit trails;
- regulatory and accounting review;
- continuous performance monitoring.