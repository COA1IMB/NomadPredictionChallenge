import h2o
import pandas

h2o.init(nthreads=-1, strict_version_check=True)
dfP = pandas.read_csv("test.csv")

modelF1 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/FModels/F1")
modelF2 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/FModels/F2")
modelF3 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/FModels/F3")
modelF4 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/FModels/F4")
modelF5 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/FModels/F5")
modelB1 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/BModels/B1")
modelB2 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/BModels/B2")
modelB3 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/BModels/B3_model_20")
modelB4 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/BModels/B4")
modelB5 = h2o.load_model("/Users/Fabian/PycharmProjects/Material/BModels/B5")


#dfP.fillna(0)
df = h2o.H2OFrame(dfP)
#df.insert_missing_values(fraction=0.5, seed=1)
#dfP.to_csv("check.csv")

id = dfP["id"]


predictionF1 = modelF1.predict(df)
predictionF2 = modelF2.predict(df)
predictionF3 = modelF3.predict(df)
predictionF4 = modelF4.predict(df)
predictionF5 = modelF5.predict(df)
predictionB1 = modelB1.predict(df)
predictionB2 = modelB2.predict(df)
predictionB3 = modelB3.predict(df)
predictionB4 = modelB4.predict(df)
predictionB5 = modelB5.predict(df)

probf = predictionF1 + predictionF2 + predictionF3 + predictionF4 + predictionF5
probb = predictionB1 + predictionB2 + predictionB3 + predictionB4 + predictionB5

probf = probf/5
probb = probb/5

predictionFrameF = h2o.as_list(probf, use_pandas=True)
predictionFrameB = h2o.as_list(probb, use_pandas=True)


frames = [id,predictionFrameF, predictionFrameB]
result = pandas.concat(frames, axis=1)
result.columns = ['id','formation_energy_ev_natom', 'bandgap_energy_ev']

result.to_csv("predictions.csv",index=False)



