import pickle

# Load the pipeline and inspect it
with open("harassment_model_pipeline.pkl", "rb") as model_file:
    model_pipeline = pickle.load(model_file)

print(model_pipeline)
