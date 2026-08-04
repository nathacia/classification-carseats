# Classification Tree: Car Seat Sales Prediction

## Overview
This project builds and evaluates a decision tree classifier in R to
predict whether a car seat product will have high sales, using the
`Carseats` dataset from the ISLR package. It covers an initial full-dataset
tree, a train/test split evaluation, 10-fold cross-validation, and pruning
— with an honest comparison of results across all four stages.

## Dataset
`Carseats` (ISLR package) — 400 observations of simulated car seat sales
data, including price, competitor pricing, income, advertising, shelf
location, age, education, and urban/US market indicators.

## Preprocessing
Converted continuous `Sales` into a binary target `High` (Yes/No), using a
threshold of 8 (sales units) — dropped the original `Sales` column
afterward, since keeping both would let the tree trivially split on a
variable derived directly from the target.

## Modeling approach and results

### Initial tree (full dataset)
- 27 terminal nodes, using all 8 available predictors
- Training misclassification rate: 9%

### Train/test split tree
- 300-observation split (`set.seed(123)`, 200 training rows)
- 20 terminal nodes
- Training misclassification rate: 11.5%
- **Test accuracy: 76%** (87+65 correct out of 200)

### 10-fold cross-validation
Using `caret`'s `train()` with the `rpart` method (a different decision
tree implementation than the `tree` package used above) across a range of
complexity parameters:
- Best complexity parameter: cp = 0.0318
- **Cross-validated accuracy: ~63.3%**

This is notably lower than the single train/test split's 76% accuracy.
Since the CV step used a different underlying algorithm (`rpart`) than the
train/test tree (`tree`), this isn't a perfectly controlled comparison —
but it's a useful signal that the single-split accuracy may be optimistic,
and a more rigorous evaluation (same algorithm, cross-validated) would
give a more trustworthy performance estimate.

### Pruned tree (4 terminal nodes)
- **Test accuracy: 71.5%** (77+66 correct out of 200) — actually *lower*
  than the unpruned tree's 76%

Unlike a companion project (stroke likelihood prediction), where pruning
improved test accuracy, pruning here reduced it. This is a legitimate,
worth-reporting result: pruning trades variance for bias, and isn't
guaranteed to help on every dataset — here the simpler 4-node tree appears
to have lost real predictive signal that the fuller tree captured.

## Tools
R, `ISLR`, `tree`, `caret`

## What I'd do differently
- Run cross-validation using the same `tree` package/algorithm as the
  train/test split, rather than switching to `rpart`, for a true
  apples-to-apples comparison
- Investigate why pruning hurt performance here — possibly the `best=4`
  pruning level was too aggressive; comparing accuracy across a range of
  pruned tree sizes (not just one) would clarify the actual bias-variance
  tradeoff at play
