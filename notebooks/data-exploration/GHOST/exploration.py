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

# The GHOSTS dataset contains multiple json files and I want to combine them to one big json/csv file or dataframe that is what I will do in this jupyter notebook

import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import Latex


# +
directory = "../../../data/GHOSTS/"
all_dataframes = {}

for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            subject = "".join(str(file_path).replace(directory, '').split('.')[:-1])
            df = pd.read_json(file_path)
            all_dataframes[subject] = df

list(all_dataframes.keys())

# +
# directory = "../../../data/GHOSTS/dataset_9jan/"
# all_dataframes = {}

# for filename in os.listdir(directory):
#     if filename.endswith(".json"):
#         file_path = os.path.join(directory, filename)
#         df = pd.read_json(file_path)
#         subject = "".join(filename.split(".")[:-1])
#         all_dataframes[subject] = df
# -

for subject in all_dataframes:
    print(subject)
    column_names = all_dataframes[subject].columns.tolist()
    column_names.sort()
    print(column_names)
    print()


# ### All dataframes have identical columns, exept "Olympiad Problem Solving" so we will look into that here

all_dataframes["dataset_30jan\Olympiad Problem Solving"].head(5)

for i in all_dataframes["dataset_30jan\Olympiad Problem Solving"].iloc:
    for j in i:
        if type(j) == dict:
            print(j)

# ### We see that all prompt are empty in "Olympiad Problem Solving"

all_dataframes.keys()

# ### Now we look at the values of all dataframes
#
# lets start by checking the number of unique values in each column, note that some columns contain a list, thus the amount of unique values can not be calculated at this stage

# +
column_names = list(all_dataframes['dataset_30jan\Definition Retrieval'].columns)
column_names.insert(0, 'name')
nr_unique_values = pd.DataFrame(columns=column_names)

for df_name in all_dataframes:
    tmp_df_nr_unique_values = {'name': df_name}
    for column in all_dataframes[df_name].columns:
        try:
            tmp_df_nr_unique_values[column] = len(all_dataframes[df_name][column].unique())
        except:
            continue

    tmp_df = pd.DataFrame([tmp_df_nr_unique_values])
    nr_unique_values = pd.concat([nr_unique_values, tmp_df])

nr_unique_values = nr_unique_values.drop(columns=['errorcodes', 'warningcodes',2, 3, 4, 5, 6, 7, 8, 9])
nr_unique_values
# -

# We see that "J Munkres, Topology (ch 1)", "J Munkres, Topology (ch 2)", "R Durret, Probability Theory", "W Rudin, Functional Analysis (ch 1)", and "W Rudin, Functional Analysis (ch 2)" all have 1 unique value in the prompt column so we will look at these dataframes to see what this value is.

# +
one_upv_dfs = [
    "dataset_30jan\J Munkres, Topology (ch 1)", 
    "dataset_30jan\J Munkres, Topology (ch 2)", 
    "dataset_30jan\R Durret, Probability Theory", 
    "dataset_30jan\W Rudin, Functional Analysis (ch 1)", 
    "dataset_30jan\W Rudin, Functional Analysis (ch 2)"
]


for name in one_upv_dfs:
    print(name)
    print(all_dataframes[name]["prompt"].unique())
    print()


# -

# ### These 4 files all have no prompt
#
# -----------------------------------------------------
#
#
# Now lets look at the distribution of the usable columns per file
#

#

# +
def barplot_ratings(df, columnname, ax):
    column_counts = df[columnname].value_counts(normalize=True)
    column_counts = column_counts.reindex([1, 2, 3, 4, 5])
    column_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax

def barplot_dfcolumn_wordcount_distr(df, columnname, ax):
    column_counts = df[columnname].str.split().str.len()
    column_counts.plot(kind='hist', ax=ax, color='skyblue')
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_ylim(top=25)
    return ax

fig, axs = plt.subplots(51, 4, figsize=(12, 180))
fig.suptitle('Distribution of usable values in subsets') 
fig.tight_layout()
fig.subplots_adjust(top=0.97)

for i, (df, ax) in enumerate(zip(all_dataframes, axs)):
    try:
        barplot_dfcolumn_wordcount_distr(all_dataframes[df], 'comment', ax[0])
        barplot_ratings(all_dataframes[df], 'rating', ax[1])
        barplot_dfcolumn_wordcount_distr(all_dataframes[df], 'prompt', ax[2])
        barplot_dfcolumn_wordcount_distr(all_dataframes[df], 'output', ax[3])
        ax[0].set_ylabel(df)
        if i == 0:
            ax[0].set_title('comment length')
            ax[1].set_title('ratings (normalized)')
            ax[2].set_title('prompt length')
            ax[3].set_title('output length')
    except:
        continue


# +
for dfname in all_dataframes:
    try:
        for i, (prompt,output,rating,comment) in enumerate(zip(
                all_dataframes[dfname].head(3)['prompt'].values,
                all_dataframes[dfname].head(3)['output'].values,
                all_dataframes[dfname].head(3)['rating'].values,
                all_dataframes[dfname].head(3)['comment'].values,
                    )
                ):
            if prompt == '': continue
            if i == 0:
                print("---------------------------------- ")
                print("---------------------------------- ")
                print(f"NAME: {dfname}" )

            datapoint = str(f" prompt:  {prompt} \n\n output {output} \n\n comment {comment} \n\n rating {rating} \n\n\n\n ----" 
                            )
            
            display(Latex(datapoint))
        display(Latex("-------------"))
    except:
        continue

        
# -

# ### Now we will look at what subsets of the GHOSTS dataset are in line with high school math problems. <br> <br>
#
# We have the following datasets:
#
#
# |subset name | usable? |
# |--|--|
# | Definition Retrieval| ( Definitions are not something that dutch high school students are tested on )|
# | J Munkres, Topology (ch 1)| ( No questions available ) |
# | J Munkres, Topology (ch 2)| ( No questions available ) |
# | MATH Algebra| ( $\textbf{This is something that dutch high school students are tested on}$ )|
# | MATH Counting and Probability| ( $\textbf{This is something that dutch high school students are tested on}$ )|
# | MATH Prealgebra| ( $\textbf{This is something that dutch high school students are tested on}$ )|
# | MATH Precalculus| ( $\textbf{This is something that dutch high school students are tested on}$ )|
# | Named Theorem Proof Completion| ( Proofs are not something that dutch high school students are tested on )|
# | Olympiad Problem Solving| ( No questions available )|
# | Proofs Collection A| ( Proofs are not something that dutch high school students are tested on )|
# | Proofs Collection B Prealgebra| ( Proofs are not something that dutch high school students are tested on )|
# | Proofs Collection B Precalculus| ( Proofs are not something that dutch high school students are tested on )|
# | R Durret, Probability Theory| ( No questions available )|
# | Reverse Definition Retrieval| ( Definitions are not something that dutch high school students are tested )|
# | Symbolic Integration	100| ( $\textbf{This is something that dutch high school students are tested on}$ )|
# | W Rudin, Functional Analysis (ch 1)| ( No questions available )|
# | W Rudin, Functional Analysis (ch 2)| ( No questions available )|

# This gives a total of $\textbf{100 + 50 + 50 + 18 + 20 = 238}$ (question, answers, ratings) triplets from the GHOSTS9jan dataset that can be used for this project. For the GHOSTS30jan dataset we also have $\textbf{100 + 50 + 50 + 18 + 20 = 238}$ (question, answers, ratings) triplets. And the miniGHOSTS_gpt4 dataset contains $\textbf{10 + 10 + 10 + 10 + 10 = 50}$ (question, answers, ratings) triplets <br><br><br><br><br>
# Which gives a total of $\textbf{526}$ potential usable datapoints.
