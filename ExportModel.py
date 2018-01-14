import h2o

h2o.init(nthreads=-1, strict_version_check=True)

modelF1 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/B1_model_175")