# ADVISE Biomass Model Analysis

## Overview

This repository contains a Jupyter notebook demonstrating an application of **ADVISE (Analytical Derivative Visualization & Interpretation for Scientific Explanation)** to scientific regression and symbolic equation discovery.

The example uses tree-level observations of **aboveground biomass (AGB)** as the response variable and **tree height (H)** and **diameter at breast height (DBH)** as predictor variables. The notebook is intended as an application example of the ADVISE workflow and can be adapted to other scientific datasets.

The general workflow is:

1. Prepare a scientific regression dataset.
2. Train a deep neural network (DNN).
3. Optimize DNN hyperparameters.
4. Calculate observation-level partial derivatives using automatic differentiation.
5. Evaluate derivative consistency and reliability.
6. Use derivative information to guide symbolic regression.
7. Evaluate candidate equations using prediction error and derivative agreement.
8. Validate the resulting equation on independent observations.

> **Important:** The workflow should be interpreted as a hypothesis-generation and model-selection procedure. Agreement with DNN-derived derivatives does not by itself establish a unique governing equation or causality.

## Notebook

**Recommended filename**

`ADVISE_Biomass_Model_Analysis.ipynb`

Although the notebook demonstrates biomass modeling, the workflow is designed to be reusable for other scientific regression problems.

## Requirements

Major Python packages used by the notebook include:

- NumPy
- pandas
- scikit-learn
- PyTorch
- Optuna
- PySR
- Matplotlib
- Plotly
- SciPy
- Cartopy
- joblib

A typical environment can be prepared with:

```bash
pip install numpy pandas scikit-learn scipy matplotlib plotly torch optuna pysr cartopy joblib
```

Depending on the PySR/Julia configuration, additional Julia setup may be required.

## Input Data

The notebook uses separate training and validation datasets.

The biomass example uses:

- `H` — tree height
- `DBH` — diameter at breast height
- `AGB` — aboveground biomass

For another application, replace these variables with the appropriate predictors and response.

A simple CSV structure is:

```text
x1,x2,x3,...,y
...
```

For example:

```text
height,dbh,biomass
18.2,24.1,185.4
22.7,31.8,302.7
```

Before analysis, check missing values, units, outliers, physical plausibility, predictor distributions, and sampling density.

## Configuration

The application-specific settings should be defined near the beginning of the notebook. For example:

```python
DATA_PATH = Path("data/training.csv")
VALID_PATH = Path("data/validation.csv")
SAVE_PATH = Path("outputs")

TARGET_COLUMN = "AGB"
FEATURE_COLUMNS = ["H", "DBH"]
```

For a new application, the most important settings to change are:

1. Input-data paths.
2. Predictor-column names.
3. Response/target-column name.
4. Training/validation partition.
5. Output directory.
6. DNN architecture and hyperparameter search ranges.
7. Symbolic-regression operator set.
8. Derivative-selection or quality-control thresholds.

## ADVISE Workflow

### 1. Prepare the data

Load the training datasets and identify predictors and response.

The independent validation dataset should not be used to tune the DNN or select the final symbolic equation.

### 2. Train the DNN

The DNN represents the relationship

$$
\[
\hat{y}=f_\theta(x_1,x_2,\ldots,x_p).
\]
$$

The trained model provides both predicted responses and local partial derivatives.

A sufficiently accurate DNN is important because the derivative information extracted later reflects the behavior learned by the DNN.

### 3. Hyperparameter optimization

The notebook uses **Optuna** to search for suitable DNN hyperparameters.

Depending on the application, these may include:

- number of hidden layers;
- number of neurons;
- learning rate;
- batch size;
- dropout;
- activation functions;
- training epochs.

Do not optimize hyperparameters against the final independent validation/test dataset.

### 4. Calculate partial derivatives

Automatic differentiation is used to calculate:
$$
\[
\frac{\partial \hat{y}}{\partial x_j}.
\]
$$
These derivatives quantify the local sensitivity of the DNN prediction to each predictor.

For example,
$$
\[
\frac{\partial \hat{AGB}}{\partial DBH}
\]
$$
describes the local change in predicted biomass with respect to DBH while holding the other predictors fixed.

If predictors were standardized before DNN training, derivatives must be transformed appropriately before interpretation in physical units.

### 5. Identify reliable functional behavior

Derivative information can be used to identify observations or regions where the learned local behavior is sufficiently consistent for symbolic modeling. Because symbolic regression can discover analytical expressions from relatively small datasets, data quality and representativeness are significantly more critical than overall sample volume.

Inspect:

- predictor distributions;
- response distributions;
- contribution distributions;
- derivative distributions;
- observation density;
- signal-to-noise characteristics;

Thresholds used in the biomass example should be treated as application-specific rather than universal constants.

### 6. Symbolic regression

The notebook uses **PySR** to search for explicit mathematical expressions:

\[
y \approx g(x_1,x_2,\ldots,x_p).
\]

The available operators strongly influence what equations can be discovered. The operator set should therefore be selected using domain knowledge, expected functional forms, dimensional consistency, and numerical stability.

For example:

```text
+
-
*
/
^
exp
```

If the required functional form is excluded from the operator set, symbolic regression cannot discover it.

### 7. Derivative-guided equation evaluation

Candidate equations are evaluated using both response predictions and local derivatives.

For a candidate equation \(g(x)\):
$$
\[
\frac{\partial g}{\partial x_j}
\approx
\frac{\partial \hat{y}}{\partial x_j}.
\]
$$
This can distinguish equations that have similar prediction accuracy but different local functional behavior.

Equation selection should consider:

- prediction error;
- derivative error;
- equation complexity;
- physical plausibility;
- independent validation.

### 8. Variables absent from a candidate equation

A symbolic equation may omit an input variable. For example,
$$
\[
y = a x_1^2
\]
$$
has
$$
\[
\frac{\partial y}{\partial x_2}=0.
\]
$$
If the DNN-derived derivative with respect to `x2` is nonzero, the candidate can receive a derivative penalty.

The corresponding option is conceptually controlled by:

```python
include_unused_variables=False
```

Enable this when omitted variables with meaningful DNN-derived derivative information should be penalized.

### 9. Independent validation

Evaluate the final symbolic equation using data that were not used for DNN training, hyperparameter optimization, symbolic-model selection, or final equation selection.


## Adapting the Notebook to Another Application

The notebook can be adapted to ecological, remote-sensing, hydrological, climatic, or other scientific regression problems.

### Step 1 — Define the scientific problem

Document:

```text
Response variable:
Predictor variables:
Expected functional relationship:
Sample size:
Independent validation dataset:
```

### Step 2 — Prepare the data

Create training and independent validation datasets with consistent columns and units.

### Step 3 — Update the configuration

Change:

```python
DATA_PATH
VALID_PATH
SAVE_PATH
TARGET_COLUMN
FEATURE_COLUMNS
```

### Step 4 — Check preprocessing

Use the training data to fit preprocessing transformations and apply the same transformation to validation data:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
```

Do not fit a new scaler to the validation dataset.

### Step 5 — Train and evaluate the DNN

Verify that the DNN provides adequate predictive performance before relying on its derivatives.

### Step 6 — Inspect derivatives

For each predictor, inspect whether derivative patterns are:

- physically plausible;
- stable;
- adequately sampled;
- affected by predictor correlations;
- unstable near domain boundaries.

### Step 7 — Configure symbolic regression

Select a scientifically appropriate operator set and equation-complexity range. Start with a restricted operator set rather than unnecessarily expanding the search space.

### Step 8 — Compare equations

Do not select an equation solely because it has the lowest training error. Consider:

```text
Prediction accuracy
+
Derivative agreement
+
Equation complexity
+
Physical plausibility
+
Independent validation
```

## Outputs

Depending on notebook configuration, outputs can include:

- trained DNN models;
- prediction results;
- partial-derivative estimates;
- derivative diagnostic plots;
- variable-response plots;
- symbolic-regression candidate equations;
- derivative-based equation scores;
- validation statistics;

Keep outputs separate from input data to improve reproducibility.

## Computational Considerations

The DNN training and derivative stages are generally much less expensive than symbolic regression. In the original biomass application, symbolic regression was the dominant computational step.

For exploratory analyses:

1. Test the workflow on a small dataset.
2. Use a restricted operator set.
3. Limit the symbolic-search budget.
4. Inspect candidate equations.
5. Run the full symbolic search only after the workflow is verified.

Avoid accidentally launching a multi-hour PySR search during notebook development.

## Scientific Interpretation and Limitations

The ADVISE workflow should be interpreted as a **hypothesis-generation and model-selection framework**, rather than an automatic mechanism-discovery system.

Important limitations include:

1. **DNN dependence** — derivative information comes from the trained DNN and can be unreliable if the DNN poorly represents the underlying relationship.
2. **Noise sensitivity** — measurement and predictor noise can reduce prediction and derivative fidelity.
3. **Sampling distribution** — sparse regions may produce less reliable local derivatives.
4. **Correlated predictors** — strong correlations can make individual-variable sensitivities difficult to identify uniquely.
5. **Symbolic search limitations** — results depend on operators, complexity limits, search settings, and computational budget.
6. **Functional-form limitations** — an equation cannot be recovered if its required functional form is excluded from the search space.
7. **Extrapolation** — DNN derivatives and symbolic equations may behave poorly outside the observed domain.
8. **Causality** — prediction and derivative agreement do not by themselves establish causality.
9. **Uniqueness** — multiple equations can produce similar predictions and derivative agreement over the available observations.

Independent validation and domain-specific scientific knowledge should remain central to interpretation.

The key principle is to treat the notebook as a **general scientific analysis framework**, not as a fixed biomass model. The biomass example demonstrates how the ADVISE workflow can be transferred to another scientific regression problem.
