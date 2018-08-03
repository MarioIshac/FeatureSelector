import numpy as np

def new_model_constructor(model_class, model_fitter_name):
    def construct_model(observed_input, observed_output):
        model = model_class()

        model_fitter = getattr(model, model_fitter_name)
        model_fitter(observed_input, observed_output)

        return model

    return construct_model
