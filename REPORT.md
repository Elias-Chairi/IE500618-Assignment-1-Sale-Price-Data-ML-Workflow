# Part A

**Answer Q1-Q7 in the written submission. Your Jupyter Notebook should contain the code, figures, tables and short reasoning that
support the answers. Refer to evidence from your notebook where appropriate.**

- **Q1.** What were the most important things you learned from exploring the dataset? Refer to relevant statistics, missing values, distributions, unusual observations and visualizations.

Each row is one house sale in Ames, Iowa (2006-2010); the target is `SalePrice` in dollars (`01_eda` 1.1).

1. **The target is right-skewed** (`01_eda` 1.3): \$12,789-\$755,000, median \$160,000, mean
   \$180,796, skew 1.74 (-0.01 after log), so expensive houses dominate a squared-error fit.
2. **`NaN` has two meanings** (`01_eda` 1.4). Often it means *absent* (no fireplace → `NaN` in
   `Fireplace Qu`); sometimes *unrecorded* (`Lot Frontage`). All 157 houses missing every garage
   category have `Garage Area == 0`, and 1,745 of 1,775 missing `Mas Vnr Type` values have
   `Mas Vnr Area == 0`, so dropping high-`NaN` columns would discard real facts.
3. **Some numbers are not measurements** (`01_eda` 1.1-1.2). `MS SubClass` is a code (price does not
   move monotonically with it); `Order` and `PID` are identifiers. `PID` correlates -0.25 with price
   only as a location proxy.
4. **Outliers are a different kind of sale** (`01_eda` 1.7). Three houses over 4,000 sq ft are
   `Partial` sales and sold far below trend (one: 5,642 sq ft, \$160,000), a different pricing
   process rather than typos. The other two (\$745,000, \$755,000) sit on the trend.
5. **Scales differ widely** (ratings 1-10 vs lot areas in tens of thousands), so we standardise.

- **Q2.** Which features appeared to be useful for predicting house prices? What evidence led you to this conclusion? Refer to relevant visualizations, descriptive statistics, correlations or your understanding of the variables.

The strongest numerical predictors measure **size** and **quality** (`01_eda` 1.5):

| Feature | r with `SalePrice` |
|---|---|
| `Overall Qual` | 0.80 |
| `Gr Liv Area` | 0.71 |
| `Garage Cars` / `Garage Area` | 0.65 / 0.64 |
| `Total Bsmt SF` / `1st Flr SF` | 0.63 / 0.62 |
| `Year Built` / `Year Remod/Add` | 0.56 / 0.53 |

Plots confirm this (`01_eda` 1.6): median price rises at every step of `Overall Qual`, and price
rises with living area, apart from the large `Partial` sales (1.7). Among categoricals,
`Neighborhood` medians range from \$88,250 to \$319,000 (1.8), and for the four rating columns plotted
(`Po < Fa < TA < Gd < Ex`) median price generally rises with each step up the scale (1.8).

- **Q3.** How did you create the training and test sets? Why should the test set be kept separate during model development, and how does this help provide an unbiased evaluation on unseen data?

We made a **stratified 80/20 split** on `Overall Qual` (`random_state=42`, `02_modeling` Step 2), the
strongest predictor, so both sets have the same quality mix. Sparse tails were merged into six bands
(`1-4`, `5`, `6`, `7`, `8`, `9-10`). This gave 2,344 training and 586 test houses.

**Why keep the test set separate**: it estimates performance on unseen houses, which is only honest
if the test data influenced nothing the models learned (imputation values, scaling statistics,
category sets, weights). Leakage would inflate the score without improving real performance. The
test set is used only for `predict` in Step 6; no model was tuned. One limitation: the exploration in
`01_eda` used the full dataset before the split. Nothing was fitted there, but its statistics include
test rows.

- **Q4.** How did you handle missing values and categorical variables? Explain your main preprocessing decisions, including scaling or transformations if used. Why should preprocessing be learned from the training data and then applied to the test data?

We used four preprocessing routes (`02_modeling` Step 3):

| Route | Imputation | Encoding | Reason |
|---|---|---|---|
| `num` (35) | median | `StandardScaler` | `NaN` is a genuine unknown |
| `ord` (20) | `"NA"` | ordinal, explicit order, then `StandardScaler` | order is information; `"NA"` = absent, below "poor" |
| `nom_absent` (5) | `"None"` | one-hot | `NaN` means the feature does not exist |
| `nom_unknown` (20) | most frequent | one-hot | `NaN` is a recording failure |

`MS SubClass` and `Mo Sold` are stored as numbers but go to the `nom_unknown` route because they are
categorical (1.2). Ordinal encoding keeps ratings in one ordered column, though it assumes equal steps.
One-hot encoders ignore unseen categories; the ordinal encoder maps them to -1. All models get the same
features and the target stays in dollars.

**Why fit on training data only**: medians, standard deviations and category sets are statistics of
the data, so computing them before splitting leaks test information. Every step sits inside a
`Pipeline`/`ColumnTransformer`, so `fit(x_train)` learns them and `predict(x_test)` only applies them.

- **Q5.** Which features did you finally use for prediction? Did you remove any features or create new ones? Explain the reasoning behind your main feature-selection or feature-engineering decisions.

**Created** (Step 4), row-wise so they cannot leak:

- `Total SF` = basement + 1st + 2nd floor area
- `Total Bath` = full baths + 0.5 × half baths (incl. basement)
- `House Age` = `Yr Sold - Year Built`; `Remod Age` = `Yr Sold - Year Remod/Add` (clipped at 0, since a few rows record a year after the sale)
- `Total Porch SF` = sum of five porch/deck areas

**Removed** (Step 3): `Order`, `PID` (identifiers); `Year Built`, `Year Remod/Add`, `Yr Sold`
(replaced by ages); `Garage Yr Blt` (159 `NaN`, 157 of them houses without a garage, plus a 2207 typo;
garage presence is captured by other garage columns).

Everything else was kept: 75 original columns plus the 5 new ones, including `Sale Condition` so models
can learn that non-normal sales differ. The three `Partial` outliers (Q1) were removed **from training
only**, by a rule based on size and sale type, never price. The test set was left untouched.

- **Q6.** What RMSE and R² values did you obtain for each of the three models on the test data? Present your results clearly in a table. Which model performed best on the test data?

Results (`02_modeling` Step 6):

| Model | Test RMSE | Test R² |
|---|---|---|
| **Linear Regression** | **\$23,924** | **0.914** |
| Decision Tree Regression | \$33,612 | 0.829 |
| Random Forest Regression | \$24,149 | 0.912 |

Linear Regression performed best, with the lowest RMSE and highest R². Random Forest is effectively
tied (about \$225 higher RMSE on one split), while the single Decision Tree is clearly worse. All
three under-predict the most expensive houses.

- **Q7.** Based on what you learned from the analysis, what could you change in the preprocessing or features to potentially improve the prediction results without changing to a different model?

Ideas:
- Log-transform the target (skew 1.74 → -0.01, 1.3), converting predictions back to dollars. Skewed
  features such as `Lot Area` could follow, but their skew was not measured.
- Target-encode `Neighborhood` instead of 28 one-hot columns (1.8), fitted inside the pipeline.
- Replace equally spaced ordinal codes with each level's median training price; the 1.8 boxplots show unequal gaps.
- Add interactions such as `Overall Qual × Total SF` (the two strongest predictors, 1.5).
- Impute `Mas Vnr Type` as `"None"` only when `Mas Vnr Area == 0`; currently all 1,775 `NaN` become
  `"None"`, including 30 rows with a veneer area or unknown area (1.4).
- Drop near-duplicate columns (`Garage Cars` / `Garage Area`, 1.5; raw areas duplicated by `Total SF`).


# Part B

**Group-work reflection. Briefly answer:**

- **Q1.** How was the work divided among group members, and what did each member contribute?

   Every member did the exercise on their own, then we sat down together and discussed each other's solutions, and then created the final one.

- **Q2.** Which important decisions were made together as a group?

   **Decisions made together**

   - Stratifying on `Overall Qual`; another idea was to stratify using `SalePrice`.
   - Features added in the feature engineering part.

- **Q3.** How did you ensure that everyone understood the complete solution, not only their own part?

   By having multiple physical meetings going through the project.

- **Q4.** Was the work distributed fairly? Explain briefly. All group members are expected to understand the complete solution.

   Yes, all members have had fair contributions to the project overall. Since every member made their own version of the assignment, everyone did roughly the same amount of work.
