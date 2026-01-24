Now that our task is complete and we have some analytical functions in `experiments.py` I'd like to shift our focus on plotting the graphs in my notebook and saving them in appropriate directory with proper naming format. We'll plot two categories of graph: 

## Comparsion across my custom models by varying the common hyper-params

* hyper-params
- max_depth
- min_samples_split
- train_sizes is a common param that can be used via `learning_curve` function.

* custom models
- dt
- rf
- et

* datasets
- iris
- wine


* evaluation metrics
- accuracy
- f1
- auroc

```
for each param in hyper-params:

    varify the param values, run experiments and plot six figures ( 3 at top, 3 at bottom) in a single plot where
    each figure contains three graphs colored in red, blue, green representing dt,rf,et performance measure for each of the evaluation metrics. 

    The final image should look like this

    top: iris dataset
        left: accuracy (models) , middle: f1 (models) , right: auroc (models)
    bottom: wine dataset
        similar 
    
```

Since we have 3 configurable params this section should generate 3 images each having 6 plots.

*each image should be save to graphs-charts/across-custom/{hyper-param name}.extension*


## Comparison between custom vs sklearn implementation for corresponding models

* custom models
- dt
- rf
- et

* sklearn models
- dt
- rf
- et

* datasets
- iris
- wine


* evaluation metrics
- accuracy
- f1
- auroc

* hyper-params for each models

```python
# =========================
# Decision Tree Hyperparameters
# =========================

DECISION_TREE_CONFIG = {
    "max_depth": None,            # None means grow until pure or min_samples_split
    "min_samples_split": 2,
    "criterion": "entropy"            # or "entropy"
}


# =========================
# Random Forest Hyperparameters
# =========================

RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt",        # sqrt, log2, or int
    "bootstrap": True
}


# =========================
# Extra Trees Hyperparameters
# =========================

EXTRA_TREES_CONFIG = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "bootstrap": False             # Extra Trees typically do not bootstrap
}
```
* train_sizes is a common param that can be used via `learning_curve` function.

```
for each m in models:
    custom_m = custom[m]
    sk_m = sklearn[m]

    for each p in hyper-params:

        **vary the values** of p and run experiments:
            note that if the value is continuous we'll go for lineplot
            else if categorical like entropy for dt, sqrt,int,log2 for rf we should go for bar plot?

            plot 3 figures for 3 evaluation metrics in a single plot where each figure contains:

                4 graphs :
                    green: custom_m(wine)
                    red: sk_m(wine)
                    blue: custom_m(iris)
                    black: sk_m (iris)
                    evaluation metrics vs p
                left(accuracy),middle(f1),right(auroc) 
            save the plot at graphs-charts/custom-sklearn/{model_name}_{hyper-param-name}.extension
```

### confusion matrix

for each d in datasets:

    plot total of six confusion matrix in the following format:
    for each m in [dt,rf,et]

    | confusion matrix custom m |
    ------------
    | confusion matrix sklearn m |

    you should fuse corresponding confusion matrix together for better visibility. You can put the other two fused confusion matrix side by side fused for a particular d
since we have two datasets , there should be toal of 12 confusion matrix from which two image should be generated

iris image:

    | confusion matrix custom dt | confusion matrix custom rf | confusion matrix custom et |
    ------------------------------------------------------------------------------------------
    | confusion matrix sklearn dt |confusion matrix sklearn rf |confusion matrix sklearn et |

wine image:
similar format

save these images at `graphs-charts/confusion-matrix/{dataset-name}.extension`

*IMPORTANT: write modular reusable functions and provide the code in a ipynb notebook friendly manner, a little heading and then code cell. The function you need are provided in experiments.py but if you need something more feel free to add them in experiements.py and import it from there using `import function_name from experiment.experiments`*