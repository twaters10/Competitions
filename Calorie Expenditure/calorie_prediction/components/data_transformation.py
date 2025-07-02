import pandas as pd
from typing import List, Union, Optional

def one_hot_encode_dataframe(
    df: pd.DataFrame,
    columns_to_encode: Union[str, List[str]],
    prefix: Optional[Union[str, List[str]]] = None,
    prefix_sep: str = '_',
    dummy_na: bool = False,
    drop_first: bool = False,
    dtype: Optional[type] = None
) -> pd.DataFrame:
    """
    Performs one-hot encoding on specified columns of a Pandas DataFrame using pd.get_dummies().

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (Union[str, List[str]]): A single column name (str) or a list
                                                    of column names (List[str]) to be one-hot encoded.
        prefix (Optional[Union[str, List[str]]]): String or list of strings, to be appended to
                                                   column names. If None, uses the column name as prefix.
        prefix_sep (str): Separator to use when appending prefix and dummy name.
        dummy_na (bool): If True, add a column to indicate NaN values.
                         If False (default), NaN values are ignored.
        drop_first (bool): Whether to get k-1 dummies out of k categorical levels by removing
                           the first level. Set to True to avoid multicollinearity for linear models.
        dtype (Optional[type]): Data type for new columns. Defaults to uint8.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns one-hot encoded.
                      The original columns are dropped.
    """
    if not isinstance(columns_to_encode, list):
        columns_to_encode = [columns_to_encode]

    # Check if all columns to encode exist in the DataFrame
    missing_columns = [col for col in columns_to_encode if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    # Use pd.get_dummies to perform the one-hot encoding
    encoded_df = pd.get_dummies(
        df,
        columns=columns_to_encode,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        drop_first=drop_first,
        dtype=dtype
    )
    return encoded_df
