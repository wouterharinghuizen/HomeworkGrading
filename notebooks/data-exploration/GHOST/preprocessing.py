# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import matplotlib.pyplot as plt
import os

interesting_subdatasets = [
    "dataset_30jan\\MATH Algebra",
    "dataset_30jan\\MATH Counting and Probability",
    "dataset_30jan\\MATH Prealgebra",
    "dataset_30jan\\MATH Precalculus",
    "dataset_30jan\\Symbolic Integration",
    "dataset_9jan\\MATH Algebra",
    "dataset_9jan\\MATH Counting and Probability",
    "dataset_9jan\\MATH Prealgebra",
    "dataset_9jan\\MATH Precalculus",
    "dataset_9jan\\Symbolic Integration",
    "miniGHOSTS_gpt4\\MATH Algebra",
    "miniGHOSTS_gpt4\\MATH Counting and Probability",
    "miniGHOSTS_gpt4\\MATH Prealgebra",
    "miniGHOSTS_gpt4\\MATH Precalculus",
    "miniGHOSTS_gpt4\\Symbolic Integration",
]

algebra = [file for file in interesting_subdatasets if file.endswith("Algebra")]
countingAndProbability = [
    file
    for file in interesting_subdatasets
    if file.endswith("Counting and Probability")
]
prealgebra = [
    file for file in interesting_subdatasets if file.endswith("Prealgebra")
]
precalculus = [
    file for file in interesting_subdatasets if file.endswith("Precalculus")
]
symbolicIntegration = [
    file
    for file in interesting_subdatasets
    if file.endswith("Symbolic Integration")
]

# ## checking NaN rows

# +
directory = "../../../data/GHOSTS/"
all_dataframes = {}


def locations_to_dfs(locations, root_directory):
    for filename in locations:
        print(filename)
        filepath = os.path.join(root_directory, filename + ".json")
        df = pd.read_json(filepath)
        nr_rows = df.shape[0]
        df = df.dropna()
        nr_rows_wo_nan = df.shape[0]
        all_dataframes[filename] = df
        print(
            f"nr of rows with at least 1 NaN value = {nr_rows - nr_rows_wo_nan} \n"
        )


locations_to_dfs(algebra, directory)
locations_to_dfs(countingAndProbability, directory)
locations_to_dfs(prealgebra, directory)
locations_to_dfs(precalculus, directory)
locations_to_dfs(symbolicIntegration, directory)
# -

# ## finding the "solution" in each dataframe

for filename in all_dataframes:
    df = all_dataframes[filename]
    print(filename)
    display(df.head(1))


# In all the MATH subdatasets the solution is in the "comment" column. Except for the 30jan prealgebra and 30jan precalculus dataset. So initially
# 1) we will create a solution column and then
# 2) will use the solution column from the 9jan dataset on the 30jan dataset, since the questions are identical, thus the solutions are as well.

# 1)


# +
def split_officialsolution_MATH(string):
    return string.split(": ")[-1]


def split_officialsolution_SI(string):
    return string.split(", LHS")[-1]


for filename in all_dataframes:
    df = all_dataframes[filename]
    print(filename)

    if "MATH" not in filename:
        all_dataframes[filename]["solution"] = all_dataframes[
            filename
        ].ref.apply(split_officialsolution_SI)
        display(df.head(1))
        continue

    all_dataframes[filename]["solution"] = all_dataframes[
        filename
    ].comment.apply(split_officialsolution_MATH)
    display(df.head(1))
# -

# 2)

# +
all_dataframes["dataset_30jan\MATH Prealgebra"]
all_dataframes["dataset_30jan\MATH Prealgebra"].drop(
    "solution", axis=1, inplace=True
)
all_dataframes["dataset_30jan\MATH Prealgebra"] = all_dataframes[
    "dataset_30jan\MATH Prealgebra"
].merge(
    all_dataframes["dataset_9jan\MATH Prealgebra"][["prompt", "solution"]],
    on="prompt",
    how="left",
)

all_dataframes["dataset_30jan\MATH Precalculus"]
all_dataframes["dataset_30jan\MATH Precalculus"].drop(
    "solution", axis=1, inplace=True
)
all_dataframes["dataset_30jan\MATH Precalculus"] = all_dataframes[
    "dataset_30jan\MATH Precalculus"
].merge(
    all_dataframes["dataset_9jan\MATH Precalculus"][["prompt", "solution"]],
    on="prompt",
    how="left",
)

display(all_dataframes["dataset_30jan\MATH Precalculus"])
display(all_dataframes["dataset_30jan\MATH Prealgebra"])

# -

# ## now we will change the list of error codes to error strings.
#
# First we see what the strings are that correspond to the code

# e1 → missing examples, or information (e.g., the user asks it what the prime divisors of 111 are, and
# it responds with 3, missing 37); this also applies, if the student ignores a part of the prompt (e.g., an
# equivalence needs to be shown, but the student shows only one direction);
#
# e2 → a few wrong/vague statements (e.g., the user asks it what the prime divisors of 30030 are16 and
# it responds with 2, 3, 5, 7, 13 (wrong); or says that 2, 3, 5, and some other numbers are prime divisors
# (vague)); it can also denote a single statement, that is slightly vague;
#
# e3 → a lot of wrong/too vague statements (e.g., the user asks it what the prime divisors of 30030 are,
# and it responds with 2, 5, 8, 12, 13, 15 (wrong); or says that 2 and many other numbers are prime
# divisors (vague)); it can also denote a single statement, that is highly vague;
#
# e4 → wrong computations (i.e., an additional error flag to disambiguate between statements that are
# of computational nature or not);
#
# e5 → denotes wrong logic or wrong flow of arguments, which we further subdivide into specific flags,
# as we prohibit the use of e5 on its own (since it would be uninformative):
# – e5_1 → the student claims that to complete a proof, statements need to be shown that are unrelated
# to the claim;
# – e5_2 → a proof step is missing;
# – e5_3 → an edge case has not been considered;
# – e5_4 → an inference step is not supported (e.g., the student claims that from A follows B, but this
# claim is not true);
# – e5_5 → circular logical argument (i.e., using the hypothesis to prove the hypothesis);
#
# e6 → the general set-up is understood, but the legal operations are not respected or misunderstood (e.g.,
# we are given a puzzle where we are only allowed to add even integers, but the student changes the rules
# and motivates the solution by allowing the addition of odd integers; or the student misunderstands an
# adjective that has multiple mathematical meanings, such as “dual”, which can mean either topological
# dual space or algebraic dual space).

# Now we change the dfs

error_code_dict = {
    "e1": "missing examples, or information (e.g., the user asks it what the prime divisors of 111 are, and \
    it responds with 3, missing 37); this also applies, if the student ignores a part of the prompt (e.g., an \
    equivalence needs to be shown, but the student shows only one direction);",
    "e2": "a few wrong/vague statements (e.g., the user asks it what the prime divisors of 30030 are16 and \
    it responds with 2, 3, 5, 7, 13 (wrong); or says that 2, 3, 5, and some other numbers are prime divisors \
    (vague)); it can also denote a single statement, that is slightly vague;",
    "e3": "a lot of wrong/too vague statements (e.g., the user asks it what the prime divisors of 30030 are, \
    and it responds with 2, 5, 8, 12, 13, 15 (wrong); or says that 2 and many other numbers are prime \
    divisors (vague)); it can also denote a single statement, that is highly vague;",
    "e4": "wrong computations (i.e., an additional error flag to disambiguate between statements that are \
    of computational nature or not);",
    "e5": "denotes wrong logic or wrong flow of arguments, which we further subdivide into specific flags, \
    as we prohibit the use of e5 on its own (since it would be uninformative)",
    "e5_1": "the student claims that to complete a proof, statements need to be shown that are unrelated \
    to the claim;",
    "e5_2": "a proof step is missing;",
    "e5_3": "an edge case has not been considered;",
    "e5_4": "an inference step is not supported (e.g., the student claims that from A follows B, but this \
    claim is not true); ",
    "e5_5": "circular logical argument (i.e., using the hypothesis to prove the hypothesis);",
    "e6": "the general set-up is understood, but the legal operations are not respected or misunderstood (e.g., \
    we are given a puzzle where we are only allowed to add even integers, but the student changes the rules \
    and motivates the solution by allowing the addition of odd integers; or the student misunderstands an \
    adjective that has multiple mathematical meanings, such as “dual”, which can mean either topological \
    dual space or algebraic dual space).",
}


def error_code_to_str(errorcodes):
    errorstring = ""
    i = 0
    for error_nr, errorcode in enumerate(errorcodes):
        errorstring += (
            str(error_nr)
            + ": "
            + error_code_dict[errorcode]
            + "               "
        )

    return errorstring


for filename in all_dataframes:
    df = all_dataframes[filename]
    df["errorstring"] = None
    df["errorstring"] = df.errorcodes.apply(error_code_to_str)
    df.drop(
        columns=[
            "errorcodes",
            "warningcodes",
            "comment",
            "msc",
            "ref",
            "confidence",
            "timestamp",
        ],
        inplace=True,
    )
    all_dataframes[filename] = df


# +
for filename in all_dataframes:
    folder, file = filename.split("\\")
    folderpath = os.path.join(directory, "preprocessed", folder)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    filepath = os.path.join(folderpath, file + ".csv")

    df = all_dataframes[filename]

    print(filename)
    display(df.head(10))
    df.to_csv(filepath)


# +
total_df = pd.DataFrame()
for filename in all_dataframes:
    df = all_dataframes[filename]
    total_df = pd.concat([total_df, df])

total_df


# -


class GHOSTSDataloader:
    def __init__(self, csv_file: str) -> None:
        """_summary_

        Args:
            csv_file (str): path to csv file with preprocessed GHOSTS data
        """
        self.df = pd.read_csv(folder_totaldf)

    def __len__(self) -> None:
        return self.df.shape[0]

    def __getitem__(self, idx) -> dict:
        row = self.df.iloc[idx]

        return {
            "prompt": row["prompt"],
            "output": row["output"],
            "rating": row["rating"],
            "solution": row["solution"],
            "errorstring": row["errorstring"],
        }
