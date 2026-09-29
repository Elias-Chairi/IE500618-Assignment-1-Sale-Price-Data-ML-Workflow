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
3. **Some numbers are not measurements** (`01_eda` 1.1-1.2). `MS SubClass` is a code; `Order` and
   `PID` are identifiers. `PID` correlates -0.25 with price only as a location proxy.
4. **Outliers are a different kind of sale** (`01_eda` 1.7). Three houses over 4,000 sq ft are
   `Partial` sales and sold far below trend (one: 5,642 sq ft, \$160,000), a different pricing
   process rather than typos.
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

Plots confirm this (`01_eda` 1.6): median price rises monotonically with `Overall Qual`, and living
area is strongly linear apart from the Q1 outliers. Among categoricals, `Neighborhood` medians range
from \$88,250 to \$319,000 (`01_eda` 1.8), and quality ratings (`Po < Fa < TA < Gd < Ex`) raise the
median at every step. Random forest importances agree: engineered `Total SF` and `House Age` rank
2nd and 3rd (`02_modeling` 6.4).

- **Q3.** How did you create the training and test sets? Why should the test set be kept separate during model development, and how does this help provide an unbiased evaluation on unseen data?

We created training and test sets to be able to have an unbiased evaluation of the model. If knowledge about the test set had leaked into the training, it would inflate the score without improving real performance.

We made a **stratified 80/20 split** on `Overall Qual` (`random_state=42`). Sparse tails were merged into six bands (`1-4`, `5`, `6`, `7`, `8`, `9-10`).

**Why keep the test set separate** (`02_modeling` 2.1): it estimates performance on unseen houses,
which is only honest if test data influenced nothing (feature choice, imputation, scaling).
Leakage inflates the score without improving real performance. Our test set is
used only in Step 6; models were compared with 5-fold CV on training data (5.2).

- **Q4.** How did you handle missing values and categorical variables? Explain your main preprocessing decisions, including scaling or transformations if used. Why should preprocessing be learned from the training data and then applied to the test data?

We used four preprocessing routes (`02_modeling` Step 3):

| Route | Imputation | Encoding | Reason |
|---|---|---|---|
| `num` (35) | median | `StandardScaler` | `NaN` is a genuine unknown |
| `ord` (20) | `"NA"` | ordinal, explicit order | order is information; `"NA"` = absent, below "poor" |
| `nom_absent` (5) | `"None"` | one-hot | `NaN` means the feature does not exist |
| `nom_unknown` (20) | most frequent | one-hot | `NaN` is a recording failure |

`MS SubClass` and `Mo Sold` even though they are numbers are past to the `nom_unknown` route, because 
the meaning of them are categorical. Ordinal encoding keeps ratings in one ordered column, though it 
assumes equal steps. Encoders ignore unseen categories. All models get the same features, the target 
stays in dollars.

**Why fit on training data only**: medians, means, standard deviations and category sets are
statistics of the data, computing them before splitting leaks test information. Every step sits
inside a `Pipeline`/`ColumnTransformer`, so `fit(X_train)` learns them and `predict(X_test)` only
applies them.

- **Q5.** Which features did you finally use for prediction? Did you remove any features or create new ones? Explain the reasoning behind your main feature-selection or feature-engineering decisions.

**Created** (Step 4), row-wise so they cannot leak:

- `Total SF` = basement + 1st + 2nd floor area
- `Total Bath` = full baths + 0.5 × half baths (incl. basement)
- `House Age` = `Yr Sold - Year Built`; `Remod Age` = `Yr Sold - Year Remod/Add` (clipped at 0)
- `Total Porch SF` = sum of five porch/deck areas

**Removed** (Step 3): `Order`, `PID` (identifiers); `Year Built`, `Year Remod/Add`, `Yr Sold`
(replaced by ages; price by sale year is flat); `Garage Yr Blt` (`NaN` for no garage, a 2207 typo).

Everything else was kept, including `Sale Condition` so models can learn that non-normal sales
differ. The three `Partial` outliers (Q1) were
removed **from training only**, using a rule based on size and sale type, never price.

- **Q6.** What RMSE and R² values did you obtain for each of the three models on the test data? Present your results clearly in a table. Which model performed best on the test data?

Results:

| Model | Test RMSE | Test R² |
|---|---|---|
| **Linear Regression** | **\$23,924** | **0.914** |
| Decision Tree Regression | \$34,994 | 0.815 |
| Random Forest Regression | \$24,162 | 0.912 |

The Linear Regression model performed the best.

- **Q7.** Based on what you learned from the analysis, what could you change in the preprocessing or features to potentially improve the prediction results without changing to a different model?

Ideas: 
- Log-transform skewed features such as `Lot Area` (1.3).
- Target-encode `Neighborhood` instead of 28 one-hot columns (1.8).
- Space ordinal levels by training-set median price (Gd→Ex is ~500× Po→Fa) (3.2).
- Add interactions such as `Overall Qual × Total SF`.
- Impute `Mas Vnr Type` as `"None"` only when `Mas Vnr Area == 0` (1.4).


# Part B

**Group-work reflection. Briefly answer:**

- **Q1.** How was the work divided among group members, and what did each member contribute?

   Every member did the exercise on their own, then we sat down together and discussed each others solutions, then to create the final one. 

- **Q2.** Which important decisions were made together as a group?

   **Decissions made together**

   - Stratifying on `Overall Qual`, another idea was to stratify using `SalePrice`,
   - Features added in the feature engineering part.

- **Q3.** How did you ensure that everyone understood the complete solution, not only their own part?

   By having multiple physical meetings going through the project.

- **Q4.** Was the work distributed fairly? Explain briefly. All group members are expected to understand the complete solution.

   Yes, all members have had fair contributions to the project overall. Since every member made their own version of the assignment, everyone did        roughly the same amount of work.
