from pyAFL.teams import CURRENT_TEAMS
from pyAFL.utils import name_utils as nu
import pandas as pd
import numpy as np

if __name__ == "__main__":

    seasons = [2025, 2024, 2023, 2022]

    t = CURRENT_TEAMS[0]

    player_database = []
    for team in CURRENT_TEAMS:
        all_players = []
        print(f"\t{team}")
        for s in seasons:
            # Find all active players in the season. Last row is the totals row so we drop
            all_active_players_in_season = team.season_stats(year=s).iloc[:-1]
            players = all_active_players_in_season[["Player"]]
            all_players.append(players)

        df = pd.concat(all_players, axis=0)
        df[["lastName", "firstName"]] = (
            df["Player"].str.split(",", n=1, expand=True).apply(lambda s: s.str.strip())
        )
        player_database.append(df.drop("Player", axis=1))

    df = (
        pd.concat(player_database, axis=0)
        .sort_values("lastName")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df.insert(0, "playerID", np.arange(10_000, 10_000 + len(df)))

    df["name"] = df["firstName"] + " " + df["lastName"]
    df["name_key"] = df["name"].apply(nu.normalize_name)
    df.to_parquet("data/dim-player/players.parquet")
