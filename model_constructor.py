def new_model_constructor(model_class, model_fitter_name, *model_args, **model_kwargs):
    """
    :param model_class: Literal class of the model (for example, LinearRegression).
    :param model_fitter_name: Name of the instance method of said class that fits an instance of a model
                              given 2, parameters input and output.
    :param model_args: Non-keyword args to be used in constructor call of model.
    :param model_kwargs: Keyword args to be used in constructor call of model.

    :return: a function that creates, fits and returns a model given input and output to train said model on.
    """

    def construct_model(observed_input, observed_output):
        """
        :param observed_input:
        :param observed_output:

        :return: A model of type model_class constructed with non-keyword arguments model_args and keyword
                 arguments model_kwargs that is fitted based on observed_input and observed_output through
                 the instance method with name model_fitter_name.
        """

        model = model_class(*model_args, **model_kwargs)

        model_fitter = getattr(model, model_fitter_name)
        model_fitter(observed_input, observed_output)

        return model

    return construct_model


def new_custom_model_constructor(model_class, model_fitter, *model_args, **model_kwargs):
    """
    Equivalent to new_model_constructor with exception of parameter model_fitter. See new_model_constructor
    for this function's, parameters' other than model_fitter and return value's information.

    Prefer this function over new_model_constructor in cases where fitting a model cannot be done through
    one function call taking an input and output to train said model on.

    :param model_fitter: Instead of being the name of an instance method, this is an externally defined
                         function that takes a model, input and output. Intent is to fit the provided model
                         with provided input and output.
    """

    def construct_model(observed_input, observed_output):
        """
        Equivalent to construct_model of new_model_constructor, with the exception of fitting constructed
        model with external function rather than internal instance function.
        """

        model = model_class(*model_args, **model_kwargs)

        model_fitter(model, observed_input, observed_output)

        return model

    return construct_model
