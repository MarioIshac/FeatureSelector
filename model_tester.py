import numpy as np

epsilon = 1e-8

from enum import Enum

class ScoreDirection(Enum):
    HIGHER = 0
    LOWER = 1

def new_model_tester(model_predicter_name, scorer, score_direction):
    """
    :param model_predicter_name: The name of the instance method that predicts an expected output given an
                                 observed input.
    :param scorer: A function that takes an observed output, expected output and returns a score regarding the
                   effectiveness of the model that predicted the expected output (higher being worse).
    :param score_direction: The direction of a better score. Is either ScoreDirection.HIGHER or ScoreDirection.LOWER.
    :return: A model tester (a function) that takes a model, observed input, observed output and
             returns a score regarding the effectiveness of a model (higher being worse).
    """

    def test_model(model, observed_input, observed_output):
        """
        :param model: The model to score.
        :param observed_input: The input to validate the model on.
        :param observed_output: The output to validate the model on.
        :return: The effectiveness of the model, given the observed output and the expected output, which the model
                 produces given observed input.
        """
        model_predicter = getattr(model, model_predicter_name)

        expected_output = model_predicter(observed_input)

        model_score = scorer(observed_output, expected_output)

        # All wrapper method implementations assume that a higher score is better
        # If a lower score is better, take reciprocal to avoid reverse ranking
        if score_direction == ScoreDirection.LOWER:
            model_score = 1 / (model_score + epsilon)

        return model_score
    return test_model