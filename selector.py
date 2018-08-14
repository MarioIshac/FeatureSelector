import math

import numpy as np
from random import sample


class FeatureSelector():
    def __init__(self, observed_data_frame, output_column_name, cutoff=None):
        """
        :param observed_data_frame: A pandas DataFrame with input columns to consider and output column.
        :param output_column_name: Name of output column in observed_data_frame.
        :param cutoff: Number of features to select. If none, all features are selected. Regardless of
                       number of features, selected features will be sorted by significance.
        """
        self.observed_data_frame = observed_data_frame
        self.output_column_name = output_column_name
        self.input_column_names = [column_name for column_name in observed_data_frame.columns if
                                   column_name != output_column_name]

        if not cutoff:
            cutoff = len(self.input_column_names)

        self.cutoff = cutoff
        self.selected_input_column_names = []
        self.output_column = observed_data_frame[output_column_name]

        self.entry_count = len(observed_data_frame)

    def select_features(self):
        """
        Selects self.cutoff number of features, storing them in self.selected_input_column_names.
        """

        raise NotImplementedError("Instantiate one of subclasses in order to select features.")


# TODO Generalize iteration over remaining input column names in all implementation of select_next_feature for
# IterativeFeatureSelector. Currently the for loop is on the implementor's side rather than the interface's side.

class IterativeFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, cutoff=None):
        super().__init__(observed_data_frame, output_column_name, cutoff)

    def select_features(self):
        while len(self.selected_input_column_names) < self.cutoff:
            selected_input_column_name = self.select_next_feature()

            self.selected_input_column_names.append(selected_input_column_name)

    def select_next_feature(self):
        """
        :return: Returns most significant feature. May consider currently selected features (in the
                 case of wrapper method implementations), may not (in the case of filter method implementations).
        """
        raise NotImplementedError("Instantiate one of subclasses in order to select next feature.")

    def calc_remaining_input_column_names(self):
        """
        :return: Column names that have not been selected so far. This is equivalent to the column
                 names that are candidates for the next selection.
        """
        return [column_name for column_name in self.input_column_names if
                column_name not in self.selected_input_column_names]


class WrapperFeatureSelector(FeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        super().__init__(observed_data_frame, output_column_name, cutoff)

        self.model_constructor = model_constructor
        self.model_tester = model_tester
        self.train_count = math.floor(self.entry_count * train_portion)


# TODO Implement GeneticFeatureSelector

class GeneticFeatureSelector(WrapperFeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        super().__init__(observed_data_frame, output_column_name, model_constructor, model_tester, cutoff,
                         train_portion)

    def select_features(self):
        pass


class ForwardSelectionFeatureSelector(WrapperFeatureSelector, IterativeFeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        # Super in this case in WrapperFeatureSelector, and we only invoke this super constructor
        # because it is the only one that has a special procedure in itself (other super constructor
        # only forwards arguments to FeatureSelector, that class is extended for the sole reason
        # that it provides methods used for iteratively selecting features
        super().__init__(observed_data_frame, output_column_name, model_constructor, model_tester, cutoff,
                         train_portion)

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


class BackwardSelectionFeatureSelector(WrapperFeatureSelector, IterativeFeatureSelector):
    def __init__(self, observed_data_frame, output_column_name, model_constructor, model_tester, cutoff=None,
                 train_portion=0.8):
        super().__init__(observed_data_frame, output_column_name, model_constructor, model_tester, cutoff,
                         train_portion)

    def select_next_feature(self):
        highest_model_score_reduction = 0

        pre_removal_columns_names = [column_name for column_name in self.observed_data_frame.columns if
                                     column_name not in self.selected_input_column_names]
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
            post_removal_column_names = [column_name for column_name in pre_removal_columns_names if
                                         column_name != remaining_input_column_name]
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


class ChiSquaredFeatureSelector(IterativeFeatureSelector):
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
                observed_input_mask = input_column == input_class

                for output_class, output_class_count in output_classes.items():
                    expected_value = input_class_count * output_class_count / self.entry_count

                    observed_output_mask = self.output_column == output_class

                    observed_mask = observed_input_mask & observed_output_mask

                    observed_value = len(self.observed_data_frame.loc[observed_mask])

                    chi_square_test_stat_contribution = ((observed_value - expected_value) ** 2) / expected_value
                    chi_square_test_stat += chi_square_test_stat_contribution

            if chi_square_test_stat > most_significant_chi_squared_stat:
                most_significant_chi_squared_stat = chi_square_test_stat
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name


class CovarianceFeatureSelector(IterativeFeatureSelector):
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


class PearsonCorrelationCoefficientFeatureSelector(IterativeFeatureSelector):
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
                most_significant_pcc = pcc
                most_significant_input_column_name = remaining_input_column_name

        return most_significant_input_column_name
