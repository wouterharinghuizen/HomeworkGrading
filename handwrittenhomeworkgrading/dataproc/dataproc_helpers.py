# -*- coding: utf-8 -*-
"""Helper class for dataframe manipulations"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Latex, display


class GradingDataProcessor:
    """
    A class for processing and analyzing grading data stored in JSON format
    within specified directories.

    Attributes
    ----------
    data_directory : str
        The directory containing JSON files with grading data.
    problem_column_name : str
        Column name for problem descriptions.
    student_answer_column_name : str
        Column name for student answers.
    correct_answer_column_name : str
        Column name for correct answers.
    rating_column_name : str
        Column name for ratings.
    dataframes : dict
        A dictionary storing pandas DataFrames loaded from JSON files.
    """

    def __init__(
        self,
        data_directory: str,
        problem_column_name: str = "problem",
        student_answer_column_name: str = "student_answer",
        correct_answer_column_name: str = "correct_answer",
        rating_column_name: str = "rating",
    ):
        """
        Initializes the GradingDataProcessor with directory path and column
        names.

        Parameters
        ----------
        data_directory : str
            Path to the directory containing grading data in JSON files.
        problem_column_name : str, optional
            The name of the column containing problem descriptions.
            Defaults to "problem".
        student_answer_column_name : str, optional
            The name of the column containing student answers.
            Defaults to "student_answer".
        correct_answer_column_name : str, optional
            The name of the column containing the correct answers.
            Defaults to "correct_answer".
        rating_column_name : str, optional
            The name of the column containing rating scores.
            Defaults to "rating".
        """

        self.problem_column_name = problem_column_name
        self.student_answer_column_name = student_answer_column_name
        self.correct_answer_column_name = correct_answer_column_name
        self.rating_column_name = rating_column_name

        self.data_directory = data_directory
        self.dataframes = self.json_to_dataframes_in_subfolders(
            self.data_directory
        )

    def json_to_dataframes_in_subfolders(self, path: str) -> dict:
        """
        Loads JSON files as DataFrames from specified directory and its
        subdirectories.

        Parameters
        ----------
        path : str
            The path to the directory containing JSON files.

        Returns
        -------
        dict
            A dictionary of pandas DataFrames loaded from JSON files.

        Raises
        ------
        ValueError
            If any JSON file cannot be converted to a DataFrame or
            if no JSON files are found.
        """
        all_dataframes = dict()

        os_path = os.path.join(*path.split("/"))
        base_length = len(os_path)
        for root, _, files in os.walk(path):
            relative_path = root[base_length:].lstrip(os.sep)
            for filename in files:
                if filename.endswith(".json"):
                    file_path = os.path.join(root, filename)
                    subject = relative_path + "/" + filename
                    try:
                        df = pd.read_json(file_path)
                    except ValueError as ve:
                        raise ValueError(
                            f"Invalid JSON format in {file_path}: {ve}"
                        )

                    all_dataframes[str(subject)] = df

        if not all_dataframes:
            raise ValueError(
                f"No valid JSON files found in the specified directory: {path}"
            )

        return all_dataframes

    def get_names(self) -> List[str]:
        """
        Returns a list of names of the loaded DataFrames.

        Returns
        -------
        list
            A list of names of the loaded DataFrames.
        """
        return list(self.dataframes.keys())

    def __getitem__(self, identifier: int | str) -> pd.DataFrame:
        """
        Returns the DataFrame corresponding to the given index or key.

        Parameters
        ----------
        identifier : int or str
            Index or key identifying the DataFrame.

        Returns
        -------
        pd.DataFrame
            The identified DataFrame.

        Raises
        ------
        TypeError
            If the input is not an int or str.
        """
        if isinstance(identifier, int):
            key_corresponding_to_idx = list(self.dataframes)[identifier]
        elif isinstance(identifier, str):
            key_corresponding_to_idx = identifier
        else:
            raise TypeError("The get method only accepts integers and strings.")

        return self.dataframes[key_corresponding_to_idx]

    def __delitem__(self, identifier: int | str) -> None:
        """
        Deletes a DataFrame identified by an index or key.

        Parameters
        ----------
        identifier : int or str
            The index or key of the DataFrame to delete.

        Raises
        ------
        IndexError
            If an integer index is out of range.
        KeyError
            If a key is not found in the loaded DataFrames.
        TypeError
            If the identifier is not an integer or string.
        """
        if isinstance(identifier, int):  # If the identifier is an integer
            key_corresp_to_identifier = list(self.dataframes)[identifier]
            if identifier > len(self.dataframes):
                raise IndexError("index out of range")

        elif isinstance(identifier, str):  # If the identifier is a string
            key_corresp_to_identifier = identifier
            if key_corresp_to_identifier not in self.dataframes:
                raise KeyError(f"{identifier} not in the imported dataframes.")

        else:
            raise TypeError("input must be a string or integer")

        del self.dataframes[key_corresp_to_identifier]

    def __len__(self) -> int:
        """
        Returns the number of DataFrames loaded.

        Returns
        -------
        int
            The number of DataFrames loaded.
        """
        return len(self.dataframes)

    def select_dataframes(self, subdatasets: List[str]) -> None:
        """
        Filters the loaded DataFrames based on a list of specified keys.

        Parameters
        ----------
        subdatasets : list of str
            A list of keys corresponding to the DataFrames to retain.
        """
        self.dataframes = {key: self.dataframes[key] for key in subdatasets}

    def check_columns(self) -> None:
        """
        Validates that all DataFrames contain the required columns.

        Raises
        ------
        Exception
            If any DataFrame is missing one or more required columns.
        """
        required_columns = [
            self.problem_column_name,
            self.student_answer_column_name,
            self.rating_column_name,
            self.correct_answer_column_name,
        ]

        for i, df in enumerate(self.dataframes.values()):
            # Check if all required columns are in the DataFrame
            if not all(column in df.columns for column in required_columns):
                missing_columns = [
                    column
                    for column in required_columns
                    if column not in df.columns
                ]
                raise KeyError(
                    f"DataFrame number '{i}' is missing the following columns:"
                    f"{missing_columns}, you can either delete this dataframe "
                    "or use the configure_and_transform function to modify "
                    "this dataframe."
                )

    def configure_and_transform(self, transformations: dict) -> None:
        """
        Applies specified transformations to each DataFrame to configure or
        modify its structure.

        This method supports renaming of existing columns or the splitting of
        column values based on a delimiter.
        Transformations are specified via a dictionary, where each key-value
        pair corresponds to a DataFrame and its
        intended transformation.

        Parameters
        ----------
        transformations : dict
            A dictionary specifying the transformations to apply.
            Each key is a DataFrame name, and the value is either a
            string (for renaming a column) or a dictionary specifying
            a split operation with keys 'source_column' and 'delimiter'.

        Raises
        ------
        ValueError
            If an invalid DataFrame name is specified.
        KeyError
            If a specified column for renaming or splitting does not
            exist in the DataFrame.
        """
        for df_name, transform_spc in transformations.items():
            if df_name not in self.dataframes:
                raise KeyError(f"'{df_name}' is not a valid dataframe name.")

            df = self.dataframes[df_name]

            if isinstance(transform_spc, str):
                # Rename column operation
                if transform_spc in df.columns:
                    df.rename(
                        columns={
                            transform_spc: self.correct_answer_column_name
                        },
                        inplace=True,
                    )
                else:
                    raise KeyError(
                        f"Column '{transform_spc}' not found in dataframe"
                        f"'{df_name}'."
                    )
            elif isinstance(transform_spc, dict) and "split" in transform_spc:
                # Split column operation
                split_details = transform_spc["split"]
                source_column = split_details.get("source_column")
                delimiter = split_details.get("delimiter", ":")

                if source_column not in df.columns:
                    raise KeyError(
                        f"Column '{source_column}' not found in dataframe "
                        f"'{df_name}'."
                    )

                df[self.correct_answer_column_name] = df[source_column].apply(
                    lambda x: x.split(delimiter)[1] if delimiter in x else x
                )
            else:
                raise ValueError("Invalid transformation specification.")

            self.dataframes[df_name] = df

    def analyze_unique_values(
        self, print_empty_column: bool = False
    ) -> pd.DataFrame:
        """
        Analyzes and summarizes the unique values in each column across all
        DataFrames.

        This method creates a summary DataFrame that lists the count of unique
        values for each column in every loaded DataFrame. Optionally, it can
        also print out columns that are entirely empty.

        Parameters
        ----------
        print_empty_column : bool, optional
            If True, print columns that contain only one unique value,
            specifically if that unique value is an empty string. Defaults to False.

        Returns
        -------
        pd.DataFrame
            A summary DataFrame with each row representing a DataFrame
            loaded into the class, and columns representing the count of
            unique values for each column within those DataFrames.
        """
        nr_unique_values = pd.DataFrame()
        for i, df_name in enumerate(self.dataframes):
            tmp_df_nr_unique_values = {"name": df_name}
            df_contains_single_unique = False
            for column in self.dataframes[df_name].columns:
                try:
                    unique_values = self.dataframes[df_name][column].unique()
                    unique_count = len(unique_values)
                    tmp_df_nr_unique_values[column] = unique_count

                    # Check if print_single_unique is True and if the column
                    # has exactly one unique value
                    if (
                        print_empty_column
                        and unique_count == 1
                        and unique_values[0] == ""
                    ):
                        df_contains_single_unique = True
                        print(
                            f"DataFrame 'Dataframe number {i}', Column "
                            f"'{column}' is empty"
                        )
                        print()

                except Exception as e:
                    print(
                        f"Error processing DataFrame '{df_name}', Column "
                        f"'{column}': {e}"
                    )
                    continue

            if df_contains_single_unique:
                print()

            tmp_df = pd.DataFrame([tmp_df_nr_unique_values])
            nr_unique_values = pd.concat(
                [nr_unique_values, tmp_df], ignore_index=True
            )

        return nr_unique_values

    def plot_distributions(self) -> None:
        """
        Plots distributions of problem lengths, student answer lengths,
        rating distributions, and correct answer lengths for each DataFrame.

        Before plotting, this method checks the configuration of each
        DataFrame to ensure they contain the required columns. It then plots
        histograms for problem lengths, student answer lengths, and correct
        answer lengths, and a bar chart for the distribution of ratings
        within each DataFrame.

        Assumes that ratings are categorical or ordinal data that can be
        directly counted and plotted.
        """
        self.check_columns()
        num_rows = len(self.dataframes)
        fig, axs = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

        # Ensure axs is always a 2D array for consistency
        if num_rows == 1:
            axs = np.reshape(axs, (1, -1))

        for idx, (df_name, df) in enumerate(self.dataframes.items()):
            # Calculate string lengths
            problem_lengths = df[self.problem_column_name].str.len()
            student_answer_lengths = df[
                self.student_answer_column_name
            ].str.len()
            correct_answer_lengths = df[
                self.correct_answer_column_name
            ].str.len()  # New line for correct answer lengths

            # Determine bins for length distributions
            max_length = max(
                problem_lengths.max(),
                student_answer_lengths.max(),
                correct_answer_lengths.max(),
            )  # Include correct_answer_lengths
            bins = np.linspace(
                0, max_length, num=20
            )  # 20 bins for equal interval lengths

            # Plot distributions
            axs[idx, 0].hist(
                problem_lengths, bins=bins, color="skyblue", edgecolor="black"
            )
            axs[idx, 1].hist(
                student_answer_lengths,
                bins=bins,
                color="lightgreen",
                edgecolor="black",
            )
            axs[idx, 2].bar(
                df[self.rating_column_name].value_counts().index,
                df[self.rating_column_name].value_counts().values,
                color="salmon",
                edgecolor="black",
            )
            axs[idx, 3].hist(
                correct_answer_lengths,
                bins=bins,
                color="orchid",
                edgecolor="black",
            )  # Plot for correct answer lengths

            # Set titles
            axs[idx, 0].set_title(f"{df_name} - Problem Length")
            axs[idx, 1].set_title(f"{df_name} - Student Answer Length")
            axs[idx, 2].set_title(f"{df_name} - Ratings Distribution")
            axs[idx, 3].set_title(
                f"{df_name} - Correct Answer Length"
            )  # Title for correct answer lengths plot

            # Set x and y labels for all plots
            for j in range(4):  # Adjusted loop to include the new plot
                axs[idx, j].set_xlabel("Length" if j < 3 else "Rating")
                axs[idx, j].set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def combine_dataframes(
        self, identifiers: list, new_dataframe_name: str
    ) -> None:
        """
        Combines specified DataFrames into a new one and updates the loaded
        DataFrames.

        Parameters
        ----------
        identifiers : list of int or str
            Identifiers (keys or indices) of the DataFrames to combine.
        new_dataframe_name : str
            Name for the combined DataFrame.

        Raises
        ------
        ValueError
            If identifiers are not all strings or all integers, or if a
            specified DataFrame is not found.
        """
        # Convert indices to names if identifiers are integers
        if all(isinstance(id, int) for id in identifiers):
            names = [list(self.dataframes.keys())[id] for id in identifiers]
        elif all(isinstance(id, str) for id in identifiers):
            names = identifiers
        else:
            example_int = next(
                (id for id in identifiers if isinstance(id, int)), None
            )
            example_str = next(
                (id for id in identifiers if isinstance(id, str)), None
            )
            raise ValueError(
                f"Identifiers must be all strings or all integers. Found mixed"
                f" types (e.g., integer: {example_int}, string: '{example_str}"
                "'). Please ensure all identifiers are of the same type."
            )

        # Verify that all specified names exist in dataframes
        for name in names:
            if name not in self.dataframes:
                raise ValueError(
                    f"DataFrame named '{name}' not found in" " dataframes."
                )

        # Combine specified DataFrames
        combined_dataframe = pd.concat(
            [self.dataframes[name] for name in names],
            ignore_index=True,
        )

        # Store the combined DataFrame under the new name
        self.dataframes[new_dataframe_name] = combined_dataframe

        # Remove the original DataFrames
        for name in names:
            del self.dataframes[name]

    def display_data_as_latex(
        self, number_of_problems_per_dataframe: int = 3
    ) -> None:
        """
        Displays problems and student answers in LaTeX format for analysis.

        Parameters
        ----------
        number_of_problems_per_dataframe : int, optional
            Number of problems and answers to display from each DataFrame.
        """
        display(
            f"Showing the first {number_of_problems_per_dataframe} problems "
            "and answers in all 'interesting dataframes'"
        )
        for df_name, df in self.dataframes.items():
            display(df_name)
            problems = df[self.problem_column_name].to_list()[
                :number_of_problems_per_dataframe
            ]
            answers = df[self.student_answer_column_name].to_list()[
                :number_of_problems_per_dataframe
            ]

            # Combine problems and answers into one list for easier processing
            combined_texts = problems + answers

            # Format and display each item in LaTeX
            for text in combined_texts:
                # Basic LaTeX formatting - escape special characters, etc.
                # This is a simple example. Depending on your data, you may
                # need more sophisticated formatting.
                latex_safe_text = text.replace("_", "\\_")
                display(Latex(latex_safe_text))

    def display_dataframe_heads(self, n: int = 1) -> None:
        """
        Displays the first 'n' rows of each DataFrame in the class.

        Parameters
        ----------
        n : int, optional
            The number of rows from the start of each DataFrame to
            display. Defaults to 1.
        """
        for i, (df_name, df) in enumerate(self.dataframes.items()):
            print(f"[{i}] Head of DataFrame '{df_name}':")
            display(df.head(n))
            print("\n")

    def save_dfs(self, file_location: str, format: str = "csv") -> None:
        """
        Puts all dataframes into a folder.

        Parameters
        ----------
        file_location : str
            A string specifying the folder where the files should be saved.
        format : str, optional
            A string specifying the extension of the saved files, e.g., ".csv".
            Defaults to "csv".
        """
        self.check_columns()

        if format not in ["csv", "json"]:
            raise ValueError("Format must be either 'csv' or 'json'")

        if format == "csv":
            for df_name in self.dataframes:
                self.dataframes[df_name].to_csv(
                    file_location + df_name + ".csv", sep=",", encoding="utf-8"
                )

        elif format == "json":
            for df_name in self.dataframes:
                self.dataframes[df_name].to_json(
                    file_location + df_name + ".json", sep=",", encoding="utf-8"
                )
