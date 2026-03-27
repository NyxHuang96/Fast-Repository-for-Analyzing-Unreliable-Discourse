# Preprocessing Script Documentation

**Author:** Tianhao Cao  
**Date:** 2026-02-20  
**Course:** COLX 523
**Disclaimer:** This script was generated with the assistance of Gemini 3.1 Pro.

## Overview
This document describes the data preprocessing algorithm implemented in `src/Sprint 1/preprocess.py`. The script processes the raw Spam/Ham/Phish email dataset into a cleaned, mapped, and deduplicated corpus ready for further analysis or model training.

## Prerequisites
- **pandas**: Used for data manipulation. ([Documentation](https://pandas.pydata.org/docs/))
- **kagglehub**: Used for fetching datasets directly from Kaggle. ([Documentation](https://github.com/Kaggle/kagglehub/blob/main/README.md))
- **re**: Used for regular expression string operations.
- **os / pathlib**: Used for cross-platform file path management and directory creation.

## Step-by-Step Algorithm

### 1. Setup Data Paths and Directories
The script dynamically resolves absolute paths using the `pathlib.Path` library based on the current file's location. It ensures that the output directory `data/processed` exists using `os.makedirs(exist_ok=True)`.

### 2. Download and Load Raw Dataset
Using the `kagglehub.load_dataset()` method with the `KaggleDatasetAdapter.PANDAS` adapter, the script fetches the target dataset `"akshatsharma2/the-biggest-spam-ham-phish-email-dataset-300000"` directly and securely into a pandas DataFrame object.
- **Relevant documentation**: [kagglehub PANDAS adapter](https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas)

### 3. Initial Inspection
The script prints the initial shape of the dataset and displays the first 5 records using `df.head()` to allow for visual inspection of the loaded raw contents.
- **Relevant documentation**: [pandas.DataFrame.head](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)

### 4. Handle Missing Values
The script drops any rows containing `NaN` (missing values) in either the `text` or `label` columns, utilizing the `dropna(subset=["text", "label"])` method. The script calculates and logs the exact number of removed invalid rows.
- **Relevant documentation**: [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)

### 5. Handle Duplicates
To prevent data leakage, bias, and redundancy in the model training step, the script removes duplicate entries across the `text` column using `drop_duplicates(subset=["text"])`. The total count of dropped duplicate rows is logged to the terminal.
- **Relevant documentation**: [pandas.DataFrame.drop_duplicates](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html)

### 6. Map Numeric Labels to Text Categories
According to the original dataset description provided in `resources.md`, the numeric labels are encoded as follows:
- `0` -> `Ham`
- `1` -> `Phish`
- `2` -> `Spam`

The script securely maps these integer IDs to string-based categorical representations in the `label` column using the `map()` method, preventing ambiguity in classification tasks.
- **Relevant documentation**: [pandas.Series.map](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html)

### 7. Basic Text Cleaning
The script applies fundamental NLP text cleaning steps:
- **Lowercase Conversion**: All textual data is converted to lowercase characters using `str.lower()`. ([pandas.Series.str.lower](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.lower.html))
- **Whitespace Normalization**: Multiple consecutive whitespace characters (such as multiple spaces, tabs, or newlines) are reduced down to a single space, and leading/trailing spaces are stripped out. This procedure uses Python's `re.sub(r"\s+", " ", str(x)).strip()` mapped across the dataset via an `apply()` function. ([re.sub](https://docs.python.org/3/library/re.html#re.sub))

### 8. Save the Processed Dataset
Finally, the fully cleaned, deduplicated, and mapped DataFrame is saved out to `data/processed/corpus.json` without writing the explicit pandas row indices, using `to_json(index=False)`.
- **Relevant documentation**: [pandas.DataFrame.to_json](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html)

## Output
Executing the `preprocess.py` script yields:
- A checkpoint CSV file located at `data/raw/raw_df_checkpoint.csv`.
- A standardized and clean JSON corpus file located at `data/processed/corpus.json`.
- Detailed console logs outlining the real-time progress of all preprocessing stages, the quantity of filtered missing/duplicate records, and a summary of the final categorical distributions (`Ham`, `Spam`, `Phish`) inside the mapped dataset.
