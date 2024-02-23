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

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


class GHOSTSDataloader(Dataset):
    def __init__(self, csv_file: str) -> None:
        """_summary_

        Args:
            csv_file (str): path to csv file with preprocessed GHOSTS data
        """
        self.df = pd.read_csv(csv_file)

    def __len__(self) -> None:
        return self.df.shape[0]

    def __getitem__(self, idx) -> tuple:
        row = self.df.iloc[idx]

        return (
            f"(prompt: {row['prompt']}, \n output: {row['output']}, \n solution: {row['solution']})",
            {
                "rating": row["rating"],
                "errorstring": row["errorstring"],
            },
        )


file_location = os.path.join(
    "..", "..", "..", "data", "GHOSTS", "preprocessed_GHOSTS.csv"
)

# +
dataloader = GHOSTSDataloader(file_location)

dataloader_randomised = DataLoader(dataloader, batch_size=1, shuffle=True)
# -

for item in dataloader_randomised:
    display(item)
    break
