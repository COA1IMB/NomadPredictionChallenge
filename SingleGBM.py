import h2o
import numpy as np
import math

from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from numpy.ma.core import sort

import pandas

h2o.init(nthreads=-1, strict_version_check=True)

df = h2o.import_file(path="train.csv")

## pick a response for the supervised problem
formation_energy_ev_natom = "formation_energy_ev_natom"
bandgap_energy_ev = "bandgap_energy_ev"
response = formation_energy_ev_natom
gridID = 'B4'
splitSeed = 4

##Encoding categorical
spacegroup = "spacegroup"
df[spacegroup] = df[spacegroup].asfactor()

## use all other columns (except for the name & the response column ("survived")) as predictors
predictors = df.columns
del predictors[0]
del predictors[11:13]
print(predictors)

train, valid, test = df.split_frame(
    ratios=[0.6, 0.2],
    seed=splitSeed,
    destination_frames=['train.hex', 'valid.hex', 'test.hex']
)

##print(df.head)


gbm_lucky = H2OGradientBoostingEstimator(

    stopping_rounds=500,
    stopping_metric="rmsle",
    stopping_tolerance=1e-9,
    max_depth = 6,

    sample_rate=0.64,
    ntrees=10000,

    ## sample 80% of columns per split
    col_sample_rate=0.95,
    col_sample_rate_per_tree=0.75,
    col_sample_rate_change_per_level = 0.91,
    min_rows = 9,
    nbins = 	128,
    nbins_cats = 128,
    min_split_improvement = 1e-3,
    histogram_type = 'UniformAdaptive',

    ## fix a random number generator seed for reproducibility
    seed=1234,
    learn_rate=0.05,
    learn_rate_annealing=0.99,

    ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
    score_tree_interval=10)

gbm_lucky.train(x=predictors, y=response, training_frame=train, validation_frame=valid)


perf_lucky = gbm_lucky.model_performance(test)
print(perf_lucky.rmsle())

modelfile = h2o.save_model(model=gbm_lucky, path="/Users/Fabian/PycharmProjects/Material/", force=False)
print("Model saved to " + modelfile)
