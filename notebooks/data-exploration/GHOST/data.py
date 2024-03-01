# %%
import sys
from handwrittenhomeworkgrading.dataproc import GradingDataProcessor

sys.path.append("../../../")


# %%
GHOSTS = GradingDataProcessor(
    "../../../data/GHOSTS/raw_data",
    problem_column_name="prompt",
    student_answer_column_name="output",
    rating_column_name="rating",
)

# %%
for i, name in enumerate(GHOSTS.get_names()):
    print(i, name)

# %%
GHOSTS.combine_dataframes([0, 17, 34], "Definition Retrieval")
GHOSTS.combine_dataframes([0, 16, 32], "J. Munkres, Topology (ch. 1)")
GHOSTS.combine_dataframes([0, 15, 30], "J. Munkres, Topology (ch. 2)")
GHOSTS.combine_dataframes([0, 14, 28], "MATH Algebra")
GHOSTS.combine_dataframes([0, 13, 26], "MATH Counting and Probability")
GHOSTS.combine_dataframes([0, 12, 24], "MATH Prealgebra")
GHOSTS.combine_dataframes([0, 11, 22], "MATH Precalculus")
GHOSTS.combine_dataframes([0, 10, 20], "Named Theorem Proof Completion")
GHOSTS.combine_dataframes([0, 9, 18], "Olympiad Problem Solving")
GHOSTS.combine_dataframes([0, 8, 16], "Proofs Collection A")
GHOSTS.combine_dataframes([0, 7, 14], "Proofs Collection B Prealgebra")
GHOSTS.combine_dataframes([0, 6, 12], "Proofs Collection B Precalculus")
GHOSTS.combine_dataframes([0, 5, 10], "R. Durret, Probability Theory")
GHOSTS.combine_dataframes([0, 4, 8], "Reverse Definition Retrieval")
GHOSTS.combine_dataframes([0, 3, 6], "Symbolic Integration")
GHOSTS.combine_dataframes([0, 2, 4], "W. Rudin, Functional Analysis (ch. 1)")
GHOSTS.combine_dataframes([0, 1, 2], "W. Rudin, Functional Analysis (ch. 2)")

# %%
for i, name in enumerate(GHOSTS.get_names()):
    print(i, name)

# %%
GHOSTS.analyze_unique_values(print_empty_column=True)


# %%

# Indices to delete
indices_to_delete = [0, 1, 2, 12, 13, 15, 16]

# Sort indices in reverse order to avoid index shift during deletion
for index in sorted(indices_to_delete, reverse=True):
    del GHOSTS[index]

GHOSTS.get_names()


# %%
GHOSTS.display_dataframe_heads()

# %%
del GHOSTS["Olympiad Problem Solving"]

# %%
GHOSTS.display_data_as_latex()

# %% [markdown]
# ### Now we will look at what subsets of the GHOSTS dataset are in line with
# high school math problems. <br> <br>
#
# We have the following datasets: (not pushed because of linelength issues)
#
#

# %%
del GHOSTS["Named Theorem Proof Completion"]
del GHOSTS["Proofs Collection A"]
del GHOSTS["Proofs Collection B Prealgebra"]
del GHOSTS["Proofs Collection B Precalculus"]

# %%
GHOSTS.get_names()

# %%
transformations = {
    "MATH Algebra": {"split": {"source_column": "comment", "delimiter": ": "}},
    "MATH Counting and Probability": {
        "split": {"source_column": "comment", "delimiter": ": "}
    },
    "MATH Prealgebra": {
        "split": {"source_column": "comment", "delimiter": ": "}
    },
    "MATH Precalculus": {
        "split": {"source_column": "comment", "delimiter": ": "}
    },
    "Symbolic Integration": {
        "split": {"source_column": "comment", "delimiter": ", LHS"}
    },
}

GHOSTS.configure_and_transform(transformations)


# %%
GHOSTS.plot_distributions()


# %%
GHOSTS.save_dfs("../../../data/GHOSTS/tmp/")

# %%
