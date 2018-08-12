def new_model_constructor(model_class, model_fitter_name, *model_args, **model_kwargs):
    def construct_model(observed_input, observed_output):
        model = model_class(*model_args, **model_kwargs)

        model_fitter = getattr(model, model_fitter_name)
        model_fitter(observed_input, observed_output)

        return model

    return construct_model

def new_custom_model_constructor(model_class, model_fitter, *model_args, **model_kwargs):
    def construct_model(observed_input, observed_output):
        model = model_class(model_args, model_kwargs)

        model_fitter(model, observed_input, observed_output)

        return model

    return construct_model