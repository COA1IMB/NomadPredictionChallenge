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
gridID = 'F9'
splitSeed = 9


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


hyper_params_tune = {'max_depth' : list(range(3,30+1,1)),
                'sample_rate': [x/100. for x in range(5,101)],
                'col_sample_rate' : [x/100. for x in range(20,101)],
                'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                'min_rows': [2,4,6,7,8,10,12,14,16],
                'nbins': [2**x for x in range(4,11)],
                'nbins_cats': [2**x for x in range(6,13)],
                'min_split_improvement': [0,1e-8,1e-6,1e-4],
                'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}
search_criteria_tune = {'strategy': "RandomDiscrete",
                   'max_runtime_secs': 3600,  ## limit the runtime to 60 minutes
                   'max_models': 100,  ## build no more than 100 models
                   'seed' : 39435787,
                   'stopping_rounds' : 250,
                   'stopping_metric' : "RMSLE",
                   'stopping_tolerance': 1e-9
                   }

gbm_final_grid = H2OGradientBoostingEstimator(distribution='AUTO',
                                     ## more trees is better if the learning rate is small enough
                                     ## here, use "more than enough" trees - we have early stopping
                                     ntrees=10000,
                                     ## smaller learning rate is better
                                     ## since we have learning_rate_annealing, we can afford to start with a
                                     # bigger learning rate

                                     learn_rate=0.05,
                                     learn_rate_annealing = 0.99,
                                     ## learning rate annealing: learning_rate shrinks by 1% after every tree
                                     ## (use 1.00 to disable, but then lower the learning_rate)

                                     ## score every 10 trees to make early stopping reproducible
                                     # (it depends on the scoring interval)
                                     score_tree_interval=10,
                                     ## fix a random number generator seed for reproducibility
                                     seed=123,
                                     ## early stopping once the validation AUC doesn't improve by at least 0.01% for
                                     # 5 consecutive scoring events
                                     stopping_rounds=100,
                                     stopping_metric="rmsle",
                                     stopping_tolerance=1e-9)

# Build grid search with previously made GBM and hyper parameters
final_grid = H2OGridSearch(gbm_final_grid, hyper_params=hyper_params_tune,
                           grid_id=gridID,
                           search_criteria=search_criteria_tune)
# Train grid search
final_grid.train(x=predictors,
                 y=response,
                 ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
                 max_runtime_secs=3600,
                 training_frame=train,
                 validation_frame=valid)

sorted_final_grid = final_grid.get_grid(sort_by='RMSLE', decreasing=False)
best_model = h2o.get_model(sorted_final_grid.sorted_metric_table()['model_ids'][0])
performance_best_model = best_model.model_performance(test)
print(performance_best_model)
print(sorted_final_grid)

modelfile = h2o.save_model(model=best_model, path="/Users/Fabian/PycharmProjects/Material/", force=True)
print("Model saved to " + modelfile)







