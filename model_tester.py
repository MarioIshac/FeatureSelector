import numpy as np

epsilon = 1e-8

def calc_residual_sum_of_squares(expected_output, observed_output):
    errors = expected_output - observed_output
    squared_errors = np.square(errors)
    sum_squared_errors = np.sum(squared_errors)

    return sum_squared_errors

def calc_total_sum_of_squares(observed_output):
    mean_observed_output = np.mean(observed_output)
    errors_from_mean_observed_output = observed_output - mean_observed_output
    sum_errors_from_mean_observed_output = np.sum(errors_from_mean_observed_output)

    return sum_errors_from_mean_observed_output

def score_rmse(expected_output, observed_output):
    squared_errors = calc_residual_sum_of_squares(expected_output, observed_output)
    mean_squared_error = np.mean(squared_errors)
    root_mean_squared_error = np.sqrt(mean_squared_error)

    return root_mean_squared_error

def score_r2(expected_output, observed_output):
    residual_sum_of_squares = calc_residual_sum_of_squares(expected_output, observed_output)
    total_sum_of_squares = calc_total_sum_of_squares(observed_output)

    return 1 - residual_sum_of_squares / (total_sum_of_squares + epsilon)

scorers = {
    "rmse": score_rmse,
    "r2": score_r2
}



def new_model_tester(model_predicter_name, score_type):
    def test_model(model, observed_input, observed_output):
        model_predicter = getattr(model, model_predicter_name)

        expected_output = model_predicter(observed_input)
        scorer = scorers[score_type]

        model_score = scorer(expected_output, observed_output)

        return model_score
    return test_model