from pyAFL.players.models import Player
from pyAFL.teams import CURRENT_TEAMS
from pyAFL.utils import player_season_utils as psu, match_utils
import pandas as pd
from pathlib import Path

if __name__ == "__main__":

    seasons = [2025, 2024, 2023, 2022, 2021]
    match_df = pd.read_parquet("data/dim-match/matches.parquet").reset_index()
    player_df = pd.read_parquet("data/dim-player/players.parquet")

    all_players = player_df["firstName"] + " " + player_df["lastName"]

    player_database = []
    for _, row in player_df.iterrows():
        player_name = row["firstName"] + " " + row["lastName"]
        print(f"{player_name}")
        p = Player(player_name)
        all_stats = p.get_player_stats()
        # List of seasons played- each one is a dataframe of games played
        game_stats = all_stats.season_results

        all_dfs = []
        for season_df in game_stats:
            df = psu.pull_team_year_from_columns(season_df).dropna(subset="Gm")

            team_name = df["Team"].apply(match_utils.consistent_name)
            opp_name = df["Opponent"].apply(match_utils.consistent_name)

            # Alphabetical order of teams- to ignore headaches for finals/etc
            pair_sorted = [tuple(sorted(pair)) for pair in zip(team_name, opp_name)]
            df["team_1"], df["team_2"] = zip(*pair_sorted)

            all_dfs.append(df)

        player_df = pd.concat(all_dfs, axis=0).reset_index()
        player_df.insert(0, "playerID", row.playerID)
        player_df.insert(1, "name", player_name)

        # Add the gameID
        player_df = pd.merge(
            player_df,
            match_df.reset_index()[["team_1", "team_2", "Rnd", "year", "gameID"]],
            left_on=["team_1", "team_2", "Rd", "Season"],
            right_on=["team_1", "team_2", "Rnd", "year"],
        )
        player_database.append(player_df)

    df = pd.concat(player_database, axis=0)
    df["Gm"] = df["Gm"].astype(str)
    for season in seasons:
        p = Path(f"data/stats-player-match/season={season}")
        p.mkdir(exist_ok=True, parents=True)
        df.loc[df.Season == season].to_parquet(p / "stats_player_match.parquet")
