import re
import pandas as pd


def pull_team_year_from_columns(df: pd.DataFrame) -> pd.DataFrame:
    """This turns the dataframe with a multi-index for the columns into
    a flat df with the team and year added as columns.

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df  # nothing to do

    # Take the first value from the top column level, assumed like "Sydney - 2017"
    top = df.columns.get_level_values(0)[0]
    m = re.match(r"^(.*)\s*-\s*(\d{4})$", str(top))
    team, season = (m.group(1), int(m.group(2))) if m else (top, pd.NA)

    # Drop that top column level, keeping the lower-level real column names
    out = df.copy()
    out.columns = out.columns.droplevel(0)

    # Insert Team/Season as normal columns
    out.insert(0, "Team", team)
    out.insert(1, "Season", season)
    return out
