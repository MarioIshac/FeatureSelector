import math

import numpy as np
import pandas as pd
from random import sample

class FeatureSelector():
    def __init__(self, observed_data_frame, output_column_name, cutoff=None):
        self.observed_data_frame = observed_data_frame
        self.output_column_name = output_column_name
        self.input_column_names = [column_name for column_name in observed_data_frame.columns if column_name != output_column_name]

        if not cutoff:
            cutoff = len(self.input_column_names)

        self.cutoff = cutoff
        self.selected_input_column_names = []
        self.output_column = observed_data_frame[output_column_name]

        self.entry_count = len(observed_data_frame)

    def select_features(self):
        while len(self.selected_input_column_names) < self.cutoff:
            selected_input_column_name = self.select_next_feature()

            self.selected_input_column_names.append(selected_input_column_name)

    def select_next_feature(self):
        raise NotImplementedError("Instantiate one of subclasses in order to select features.")

    def calc_remaining_input_column_names(self):
        return [column_name for column_name in self.input_column_names if column_name not in self.selected_input_column_names]


class WrapperFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        """
        :param observed_data_frame: A pandas dataframe representing the data.
        :param output_column_name: The name of the output column/feature in observed_data_frame.
        :param model_constructor: A function accepting an array of input columns and the observed output column. Should
                                  return a model that is fitted to the provided data.
        :param model_tester: A function accepting the model returned by model_constructor, an array of input columns
                             and the observed output column. Should return a score representing how well the provided model
                             predicted the output, given the input columns.
        :param cutoff: The number of features to be selected. Defaults to all features (i.e, no cutoff will be performed,
                       rather the returned list will represent the ranks of each feature regarding its significance.
        :param train_portion: The portion (in decimal) of an input column that should be used to train the model.
                              The validation portion will be the remaining portion (1 - train_portion).
                              Defaults to 0.8 (80%) if not specified.
        """
        super().__init__(observed_data_frame, output_column_name, cutoff)

        self.model_constructor = model_constructor
        self.model_tester = model_tester
        self.train_count = math.floor(self.entry_count * train_portion)


class ForwardSelectionFeatureSelector(WrapperFeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None, train_portion=0.8):
        super().__init__(observed_data_frame, output_column_name, model_constructor, model_tester, cutoff, train_portion)

    def select_next_feature(self):
        highest_model_score = 0

        input_indices = list(range(len(self.observed_data_frame)))
        train_indices = sample(input_indices, self.train_count)

        for remaining_input_column_name in self.calc_remaining_input_column_names():
            test_input_column_names = self.selected_input_column_names + [remaining_input_column_name]

            test_input_columns = self.observed_data_frame[test_input_column_names]

            train_input = test_input_columns.iloc[train_indices]

            train_output = self.output_column.iloc[train_indices]

            model = self.model_constructor(train_input, train_output)

            validation_indices = [input_index for input_index in input_indices if input_index not in train_indices]

            validation_input = test_input_columns.iloc[validation_indices]
            validation_output = self.output_column.iloc[validation_indices]

            model_score = self.model_tester(model, validation_input, validation_output)

            if model_score > highest_model_score:
                highest_model_score = model_score
                input_column_name_with_highest_model_score = remaining_input_column_name

        return input_column_name_with_highest_model_score

class BackwardSelectionFeatureSelector(WrapperFeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        super().__init__(observed_data_frame, output_column_name, model_constructor, model_tester, cutoff,
                         train_portion)

    def select_next_feature(self):
        highest_model_score_reduction = 0

        pre_removal_columns_names = [column_name for column_name in self.observed_data_frame.columns if column_name not in self.selected_input_column_names]
        pre_removal_columns = self.observed_data_frame[pre_removal_columns_names]

        input_indices = list(range(len(self.observed_data_frame)))

        train_indices = sample(input_indices, self.train_count)
        validation_indices = [input_index for input_index in input_indices if input_index not in train_indices]

        pre_removal_train_input = pre_removal_columns.iloc[train_indices]
        pre_removal_train_output = self.output_column.iloc[train_indices]

        pre_removal_model = self.model_constructor(pre_removal_train_input, pre_removal_train_output)

        pre_removal_validation_input = pre_removal_columns.iloc[validation_indices]
        pre_removal_validation_output = self.output_column.iloc[validation_indices]

        pre_removal_model_score = self.model_tester(pre_removal_model, pre_removal_validation_input,
                                                    pre_removal_validation_output)

        for remaining_input_column_name in self.calc_remaining_input_column_names():
            post_removal_column_names = [column_name for column_name in pre_removal_columns_names if column_name != remaining_input_column_name]
            post_removal_columns = self.observed_data_frame[post_removal_column_names]

            post_removal_train_input = post_removal_columns.iloc[train_indices]
            post_removal_train_output = self.output_column.iloc[train_indices]

            post_removal_model = self.model_constructor(post_removal_train_input, post_removal_train_output)

            post_removal_validation_input = post_removal_columns.iloc[validation_indices]
            post_removal_validation_output = self.output_column.iloc[validation_indices]

            post_removal_model_score = self.model_tester(post_removal_model, post_removal_validation_input,
                                                         post_removal_validation_output)

            model_score_reduction = post_removal_model_score - pre_removal_model_score

            if model_score_reduction > highest_model_score_reduction:
                highest_model_score_reduction = model_score_reduction
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name


class ChiSquaredFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name):
        super().__init__(observed_data_frame, output_column_name)

    def select_next_feature(self):
        most_significant_chi_squared_stat = 0

        output_classes = self.output_column.value_counts()

        for remaining_input_column_name in self.calc_remaining_input_column_names():
            input_column = self.observed_data_frame[remaining_input_column_name]

            input_classes = input_column.value_counts()

            chi_square_test_stat = 0

            for input_class, input_class_count in input_classes.items():
                for output_class, output_class_count in output_classes.items():
                    expected_value = input_class_count * output_class_count / self.entry_count

                    observed_input_mask = input_column == input_class
                    observed_output_mask = self.output_column == output_class

                    observed_mask = observed_input_mask & observed_output_mask

                    observed_value = len(self.observed_data_frame.loc[observed_mask])

                    chi_square_test_stat_contribution = ((observed_value - expected_value) ** 2) / expected_value
                    chi_square_test_stat += chi_square_test_stat_contribution

            if chi_square_test_stat > most_significant_chi_squared_stat:
                most_significant_chi_squared_stat = chi_square_test_stat
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name


class CovarianceFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name):
        super().__init__(observed_data_frame, output_column_name)

    def select_next_feature(self):
        most_significant_covariance = 0

        for remaining_input_column_name in self.calc_remaining_input_column_names():
            input_column = self.observed_data_frame[remaining_input_column_name]

            input_output_covariance = np.cov(input_column, self.output_column)

            # Absolute Value of covariance determines how significant the relationship is between
            # input and output
            input_significance = np.abs(input_output_covariance)

            if input_significance > most_significant_covariance:
                most_significant_covariance = input_significance
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name


class PearsonCorrelationCoefficientFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name):
        super().__init__(observed_data_frame, output_column_name)

        self.output_variance = np.var(self.output_column)

    def select_next_feature(self):
        most_significant_pcc = 0

        for remaining_input_column_name in self.calc_remaining_input_column_names():
            input_column = self.observed_data_frame[remaining_input_column_name]

            input_output_covariance = np.cov(input_column, self.output_column)
            input_variance = np.var(input_column)

            # Absolute Value of covariance determines how significant the relationship is between
            # input and output
            pcc = input_output_covariance / (input_variance * self.output_variance)

            if pcc > most_significant_pcc:
                most_significant_pcc= pcc
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name

class MutualInformationFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, cutoff=None):
        super().__init__(observed_data_frame, output_column_name, cutoff)

    """@staticmethod
    def calc_entropy(data_column):
        column_entry_count = len(data_column)
        column_classes_count = data_column.value_counts()

        entropy = 0

        for column_class, column_class_count in column_classes_count.items()
            column_class_probability = column_class_count / column_entry_count

            entropy_contribution = column_class_probability * np.log2(column_class_probability)
            entropy -= entropy_contribution

    @staticmethod
    def calc_conditional_entropy(data_column, given_data_column):
        column_entry_count = len(data_column)
        given_column_entry_count = len(given_data_column)"""

    def _calc_log(self, marginal_probability_of_input_class, marginal_probability_of_output_class,
                  both_classes_joint_probability):
        pass

    def _calc_mutual_information_with_output_column(self, input_column):
        input_column_classes_count = input_column.value_counts()
        output_column_classes_count = self.output_column.value_counts()

        mutual_information = 0

        for input_column_class, input_column_class_count in input_column_classes_count.items():
            for output_column_class, output_column_class_count in output_column_classes_count.items():
                input_class_marginal_probability = 0
                output_class_marginal_probability = 0
                both_classes_joint_probability = 0

                log = 0 if both_classes_joint_probability == 0 \
                    else self._calc_log(input_class_marginal_probability, output_class_marginal_probability,
                                        both_classes_joint_probability)

                mutual_information_contribution = both_classes_joint_probability * log

                mutual_information += mutual_information_contribution

        return mutual_information

    def select_features(self):
        while len(self.selected_input_column_names) < self.cutoff:
            most_significant_mutual_information = 0
            most_significant_input_column_name = None

            for remaining_input_column_name in self.calc_remaining_input_column_names():
                remaining_input_column = self.observed_data_frame[remaining_input_column_name]

                mutual_information = self._calc_mutual_information_with_output_column(remaining_input_column)

                if mutual_information > most_significant_mutual_information:
                    most_significant_mutual_information = mutual_information
                    most_significant_input_column_name = remaining_input_column_name

            if most_significant_input_column_name == None:
                # All features have been chosen, otherwise any remaining feature would have a t-stat
                # greater than 0 and thus this wouldn't have been None
                break

            self.selected_input_column_names.append(most_significant_input_column_name)

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()

    data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    data2 = pd.read_csv("test.csv")

    print(data2.columns)

    import model_constructor as mc
    import model_tester as mt

    import sklearn.metrics.regression

    from sklearn.linear_model import LinearRegression
    from model_tester import ScoreDirection

    linear_model_constructor = mc.new_model_constructor(LinearRegression, "fit")
    mae_model_tester = mt.new_model_tester("predict", sklearn.metrics.regression.median_absolute_error, ScoreDirection.LOWER)
    ev_model_tester = mt.new_model_tester("predict", sklearn.metrics.regression.explained_variance_score, ScoreDirection.HIGHER)
    r2_model_tester = mt.new_model_tester("predict", sklearn.metrics.regression.r2_score, ScoreDirection.HIGHER)

    feature_selector = ForwardSelectionFeatureSelector(data2, "target", linear_model_constructor, mae_model_tester)
    feature_selector.select_features()
    print(feature_selector.selected_input_column_names)

    feature_selector = ForwardSelectionFeatureSelector(data2, "target", linear_model_constructor, ev_model_tester)
    feature_selector.select_features()
    print(feature_selector.selected_input_column_names)

    feature_selector = ForwardSelectionFeatureSelector(data2, "target", linear_model_constructor, r2_model_tester)
    feature_selector.select_features()
    print(feature_selector.selected_input_column_names)