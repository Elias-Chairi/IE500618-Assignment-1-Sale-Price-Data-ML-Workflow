# Part A

**Answer Q1-Q7 in the written submission. Your Jupyter Notebook should contain the code, figures, tables and short reasoning that
support the answers. Refer to evidence from your notebook where appropriate.**

- **Q1.** What were the most important things you learned from exploring the dataset? Refer to relevant statistics, missing values, distributions, unusual observations and visualizations.

Each row is one house sale in Ames, Iowa (2006-2010); the target is `SalePrice` in dollars (`01_eda` 1.1).

1. **The target is right-skewed** (`01_eda` 1.3): \$12,789-\$755,000, median \$160,000, mean
   \$180,796, skew 1.74 (-0.01 after log), so expensive houses dominate a squared-error fit.
2. **`NaN` has two meanings** (`01_eda` 1.4). Often it means *absent* (no fireplace → `NaN` in
   `Fireplace Qu`); sometimes *unrecorded* (`Lot Frontage`). All 157 houses missing every garage
   category have `Garage Area == 0`, so dropping high-`NaN` columns would discard real facts.
   Likewise, 1,745 of the 1,775 missing `Mas Vnr Type` values have `Mas Vnr Area == 0` (no veneer).
3. **Some numbers are not measurements** (`01_eda` 1.1-1.2). `MS SubClass` is a code (price does not
   move monotonically with it); `Order` and `PID` are identifiers. `PID` correlates -0.25 with price
   only as a location proxy.
4. **Outliers are a different kind of sale** (`01_eda` 1.7). Three houses over 4,000 sq ft are
   `Partial` sales and sold far below trend (one: 5,642 sq ft, \$160,000), a different pricing
   process rather than typos. The other two houses over 4,000 sq ft (\$745,000 and \$755,000) sit on
   the trend.
5. **Scales differ widely** (ratings 1-10 vs lot areas in tens of thousands, `01_eda` 1.1, 1.9), so we
   standardise.

- **Q2.** Which features appeared to be useful for predicting house prices? What evidence led you to this conclusion? Refer to relevant visualizations, descriptive statistics, correlations or your understanding of the variables.

The strongest numerical predictors measure **size** and **quality** (`01_eda` 1.5):

| Feature | r with `SalePrice` |
|---|---|
| `Overall Qual` | 0.80 |
| `Gr Liv Area` | 0.71 |
| `Garage Cars` / `Garage Area` | 0.65 / 0.64 |
| `Total Bsmt SF` / `1st Flr SF` | 0.63 / 0.62 |
| `Year Built` / `Year Remod/Add` | 0.56 / 0.53 |

Plots confirm this (`01_eda` 1.6): median price rises at every step of `Overall Qual` (1 to 10), and
price rises with living area, although the spread widens for larger houses and the three large
`Partial` sales (1.7) sit far below the trend. Among categoricals, `Neighborhood` medians range
from \$88,250 to \$319,000 (`01_eda` 1.8), and for the four quality-rating columns plotted
(`Exter Qual`, `Kitchen Qual`, `Heating QC`, `Bsmt Qual`; scale `Po < Fa < TA < Gd < Ex`) median
price generally rises with each step up the scale (1.8; the `Po` and `Fa` medians of `Kitchen Qual`
are about equal). Correlation only captures linear relationships and is a clue, not proof of
causation, which is why we also looked at the plots.

- **Q3.** How did you create the training and test sets? Why should the test set be kept separate during model development, and how does this help provide an unbiased evaluation on unseen data?

We made a **stratified 80/20 split** on `Overall Qual` (`random_state=42`, `02_modeling` Step 2),
because it is the strongest single predictor (r = 0.80), so training and test sets get the same mix
of quality levels. Sparse tails were merged into six bands (`1-4`, `5`, `6`, `7`, `8`, `9-10`) so every
stratum is large enough to split. This gave 2,344 training and 586 test houses (2,341 training houses
after removing the three outliers described in Q5).

**Why keep the test set separate**: it estimates performance on unseen houses, which is only honest
if the test data influenced nothing that the models learned (imputation values, scaling statistics,
category sets, model weights). If knowledge about the test set leaked into training, the score would
be inflated without any real improvement in performance. In our workflow the test set is only used
for `predict` in Step 6, after all models were fitted on the training data. No hyperparameter tuning
was done (default settings, only `random_state` fixed), so the test set was not used to tune the models either.

One limitation: the exploratory analysis in `01_eda` was run on the full dataset, before the split.
It was used to understand the data and choose the preprocessing, not to fit any parameter, but a
stricter workflow would run the exploration on the training set only.

- **Q4.** How did you handle missing values and categorical variables? Explain your main preprocessing decisions, including scaling or transformations if used. Why should preprocessing be learned from the training data and then applied to the test data?

We used four preprocessing routes (`02_modeling` Step 3):

| Route | Imputation | Encoding | Reason |
|---|---|---|---|
| `num` (35) | median | `StandardScaler` | `NaN` is a genuine unknown |
| `ord` (20) | `"NA"` | ordinal, explicit order, then `StandardScaler` | order is information; `"NA"` = absent, below "poor" |
| `nom_absent` (5) | `"None"` | one-hot | `NaN` means the feature does not exist |
| `nom_unknown` (20) | most frequent | one-hot | `NaN` is a recording failure |

`MS SubClass` and `Mo Sold`, even though they are stored as numbers, are passed to the `nom_unknown`
route because their meaning is categorical (`01_eda` 1.2). Ordinal encoding keeps ratings in one
ordered column, though it assumes equal steps between levels. The one-hot encoders ignore categories
that were not seen in training; the ordinal encoder maps them to -1. All models get the same features
and the target stays in dollars (not transformed).

**Why fit on training data only**: medians, means, standard deviations and category sets are
statistics of the data, so computing them before splitting leaks test information. Every step sits
inside a `Pipeline`/`ColumnTransformer`, so `fit(x_train)` learns them and `predict(x_test)` only
applies them.

- **Q5.** Which features did you finally use for prediction? Did you remove any features or create new ones? Explain the reasoning behind your main feature-selection or feature-engineering decisions.

**Created** (Step 4), row-wise so they cannot leak:

- `Total SF` = basement + 1st + 2nd floor area
- `Total Bath` = full baths + 0.5 × half baths (incl. basement)
- `House Age` = `Yr Sold - Year Built`; `Remod Age` = `Yr Sold - Year Remod/Add` (clipped at 0, because a few rows record a year after the sale year)
- `Total Porch SF` = sum of five porch/deck areas

**Removed** (Step 3): `Order`, `PID` (identifiers); `Year Built`, `Year Remod/Add`, `Yr Sold`
(replaced by ages); `Garage Yr Blt` (159 `NaN`, of which 157 are houses without a garage, plus
a 2207 typo; garage presence is already captured by `Garage Area`, `Garage Cars`, `Garage Type` and
`Garage Finish`).

Everything else was kept, so the final model uses 75 original columns plus the 5 new ones. This
includes the original area and bathroom columns next to the totals built from them, and
`Sale Condition`, so models can learn that non-normal sales differ. The three `Partial` outliers
(Q1) were removed **from training only**, using a rule based on size (over 4,000 sq ft) and sale type,
never price. The test set was left untouched so the reported score is not flattered.

- **Q6.** What RMSE and R² values did you obtain for each of the three models on the test data? Present your results clearly in a table. Which model performed best on the test data?

Results (`02_modeling` Step 6):

| Model | Test RMSE | Test R² |
|---|---|---|
| **Linear Regression** | **\$23,924** | **0.914** |
| Decision Tree Regression | \$33,612 | 0.829 |
| Random Forest Regression | \$24,149 | 0.912 |

The Linear Regression model performed the best, with the lowest RMSE and highest R². Random Forest is
close behind (about \$225 higher RMSE), while the single Decision Tree is clearly worse.

- **Q7.** Based on what you learned from the analysis, what could you change in the preprocessing or features to potentially improve the prediction results without changing to a different model?

Ideas:
- Log-transform the target: skew falls from 1.74 to -0.01 (1.3). Fit on the log scale and convert
  predictions back to dollars so RMSE stays comparable. Skewed numeric features such as `Lot Area`
  could get the same treatment, but feature skewness was not measured in the notebooks, so it would
  need to be checked first.
- Target-encode `Neighborhood` instead of 28 one-hot columns (1.8), fitted inside the pipeline so it
  does not leak.
- Replace the equally spaced ordinal codes with each level's median price in the training set. The
  Step 3 encoding assumes equal steps, but the gaps between levels in the 1.8 boxplots are visibly unequal.
- Add interactions such as `Overall Qual × Total SF`, since quality and size are the two strongest
  predictors (1.5).
- Impute `Mas Vnr Type` as `"None"` only when `Mas Vnr Area == 0` (1,745 rows, 1.4). Currently all
  1,775 `NaN` become `"None"`, including 7 houses with `Mas Vnr Area > 0` and 23 where the area is
  also missing.
- Remove or merge near-duplicate columns (`Garage Cars` / `Garage Area`, 1.5; the raw area columns
  now duplicated by `Total SF`), which may help the linear model.


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
