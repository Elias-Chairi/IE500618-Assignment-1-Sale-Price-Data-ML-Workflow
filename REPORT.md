# Part A

## Q1. What were the most important things you learned from exploring the dataset?

Each row is one house sale in Ames, Iowa (2006-2010); the target is `SalePrice` in dollars (`01_eda` 1.1).

1. **The target is right-skewed** (`01_eda` 1.3): \$12,789-\$755,000, median \$160,000, mean
   \$180,796, skew 1.74 (-0.01 after log), so expensive houses dominate a squared-error fit.
2. **`NaN` has two meanings** (`01_eda` 1.4). Often it means *absent* (no fireplace → `NaN` in
   `Fireplace Qu`); sometimes *unrecorded* (`Lot Frontage`). All 157 houses missing every garage
   category have `Garage Area == 0`, so dropping high-`NaN` columns would discard real facts.
3. **Some numbers are not measurements** (`01_eda` 1.1-1.2). `MS SubClass` is a code; `Order` and
   `PID` are identifiers. `PID` correlates -0.25 with price only as a location proxy.
4. **Outliers are a different kind of sale** (`01_eda` 1.7). Three houses over 4,000 sq ft are
   `Partial` sales and sold far below trend (one: 5,642 sq ft, \$160,000), a different pricing
   process rather than typos.
5. **Scales differ widely** (ratings 1-10 vs lot areas in tens of thousands), so we standardise.

## Q2. Which features appeared to be useful for predicting house prices?

The strongest numerical predictors measure **size** and **quality** (`01_eda` 1.5):

| Feature | r with `SalePrice` |
|---|---|
| `Overall Qual` | 0.80 |
| `Gr Liv Area` | 0.71 |
| `Garage Cars` / `Garage Area` | 0.65 / 0.64 |
| `Total Bsmt SF` / `1st Flr SF` | 0.63 / 0.62 |
| `Year Built` / `Year Remod/Add` | 0.56 / 0.53 |

Plots confirm this (`01_eda` 1.6): median price rises monotonically with `Overall Qual`, and living
area is strongly linear apart from the Q1 outliers. Among categoricals, `Neighborhood` medians range
from \$88,250 to \$319,000 (`01_eda` 1.8), and quality ratings (`Po < Fa < TA < Gd < Ex`) raise the
median at every step. Random forest importances agree: engineered `Total SF` and `House Age` rank
2nd and 3rd (`02_modeling` 6.4).

## Q3. How did you create the training and test sets?

We made a **stratified 80/20 split** on `Overall Qual` (`random_state=42`), giving 2,344 training
and 586 test rows (`02_modeling` 2.2-2.3). Sparse tails were merged into six bands (`1-4`, `5`, `6`,
`7`, `8`, `9-10`); band proportions match within 0.1 percentage points. Testing 200 seeds per
method (Appendix A) showed stratification balances bands but does **not** stabilise RMSE (F-test
p = 0.97). We kept it because it is free.

**Why keep the test set separate** (`02_modeling` 2.1): it estimates performance on unseen houses,
which is only honest if test data influenced nothing (feature choice, imputation, scaling).
Leakage inflates the score without improving real performance. Our test set is
used only in Step 6; models were compared with 5-fold CV on training data (5.2).

## Q4. How did you handle missing values and categorical variables?

We used four preprocessing routes (`02_modeling` 3.2):

| Route | Imputation | Encoding | Reason |
|---|---|---|---|
| `num` (35) | median | `StandardScaler` | `NaN` is a genuine unknown |
| `ord` (20) | `"NA"` | ordinal, explicit order | order is information; `"NA"` = absent, below "poor" |
| `nom_absent` (5) | `"None"` | one-hot | `NaN` means the feature does not exist |
| `nom_unknown` (20) | most frequent | one-hot | `NaN` is a recording failure |

`MS SubClass` and `Mo Sold` are cast to strings. Ordinal encoding keeps ratings in one ordered
column, though it assumes equal steps. Encoders ignore unseen categories. All models get the same
features; the target stays in dollars. The result is 2,341 × 259 with no missing values.

**Why fit on training data only** (3.1): medians, means, standard deviations and category sets are
statistics of the data; computing them before splitting leaks test information. Every step sits
inside a `Pipeline`/`ColumnTransformer`, so `fit(X_train)` learns them and `predict(X_test)` only
applies them.

## Q5. Which features did you finally use for prediction?

We used **80 input columns** (259 after one-hot) (`02_modeling` 4.4-4.6).

**Created** (4.1), row-wise so they cannot leak:

- `Total SF` = basement + 1st + 2nd floor area
- `Total Bath` = full baths + 0.5 × half baths (incl. basement)
- `House Age` = `Yr Sold - Year Built`; `Remod Age` = `Yr Sold - Year Remod/Add` (clipped at 0)
- `Total Porch SF` = sum of five porch/deck areas

**Removed** (4.2): `Order`, `PID` (identifiers); `Year Built`, `Year Remod/Add`, `Yr Sold`
(replaced by ages; price by sale year is flat); `Garage Yr Blt` (`NaN` for no garage, a 2207 typo).

Everything else was kept, including `Sale Condition` so models can learn that non-normal sales
differ. The three `Partial` outliers (Q1) were
removed **from training only** (4.3), using a rule based on size and sale type, never price.

## Q6. What RMSE and R² values did you obtain for each of the three models on the test data?

Results on the 586 test houses (`02_modeling` 6.2; test mean \$182,099, sd \$81,441):

| Model | Test RMSE | Test R² |
|---|---|---|
| **Linear Regression** | **\$23,924** | **0.914** |
| Decision Tree Regression | \$34,994 | 0.815 |
| Random Forest Regression | \$24,162 | 0.912 |

**Linear Regression performed best**, but it is effectively tied with Random Forest: the \$238 gap
is far below the CV standard deviations (\$1,671 and \$2,176) and the ~\$6,700 split-to-split
variation. The **Decision Tree overfits** (training RMSE \$58 vs test \$34,994); the forest's
averaging reduces this. Our split is a favourable draw (17th percentile of 200 splits).

RMSE is an aggregate: 72% of houses are within \$20,000, but the worst miss is \$152,509 (6.3).

## Q7. What could you change in the preprocessing or features to potentially improve the prediction results without changing to a different model?

**Tested: log-transform the target** (`02_modeling` 6.5) with `TransformedTargetRegressor`, so
predictions return in dollars:

| Model | RMSE raw | RMSE log | R² raw | R² log |
|---|---|---|---|---|
| Linear Regression | \$23,924 | **\$20,335** | 0.914 | **0.938** |
| Decision Tree | \$34,994 | \$34,507 | 0.815 | 0.820 |
| Random Forest | \$24,162 | \$24,615 | 0.912 | 0.908 |

It clearly helps Linear Regression, whose squared-error fit was dominated by expensive houses; trees
split on thresholds and barely change.

**Other ideas** (6.6):

- Log-transform skewed features such as `Lot Area` (1.3).
- Target-encode `Neighborhood` instead of 28 one-hot columns (1.8).
- Space ordinal levels by training-set median price (Gd→Ex is ~500× Po→Fa) (3.2).
- Add interactions such as `Overall Qual × Total SF`.
- Impute `Mas Vnr Type` as `"None"` only when `Mas Vnr Area == 0` (1.4).

Anything learned from the target must be fitted on training folds only to avoid leakage.


# Part B

**Group-work reflection. Briefly answer:**

- **Q1.** How was the work divided among group members, and what did each member contribute?

- **Q2.** Which important decisions were made together as a group?

- **Q3.** How did you ensure that everyone understood the complete solution, not only their own part?

- **Q4.** Was the work distributed fairly? Explain briefly. All group members are expected to understand the complete solution.
