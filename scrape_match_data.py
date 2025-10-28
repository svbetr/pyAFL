from pyAFL.teams import CURRENT_TEAMS
from pyAFL.utils import match_utils
import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    print("Getting data for current teams")
    seasons = [2025, 2024, 2023, 2022, 2021]
    team_df = pd.read_csv("data/dim-team/teams.csv")
    alias_df = pd.read_csv("data/dim-team/team_aliases.csv")

    team_map = dict(zip(team_df["teamName"], team_df["teamID"]))
    alias_map = dict(zip(alias_df["alias"], alias_df["teamID"]))

    games = []
    for team in CURRENT_TEAMS:
        print(f"\t{team}")
        g = team.games
        g["Team"] = team.name
        g["Date"] = pd.to_datetime(
            g["Date"], format="%a %d-%b-%Y %I:%M %p", errors="raise"
        )
        g["year"] = g["Date"].dt.year

        g["Rnd"] = g["Rnd"].str.strip("R")

        team_name = g["Team"].apply(match_utils.consistent_name)
        opp_name = g["Opponent"].apply(match_utils.consistent_name)

        # Alphabetical order of teams- to ignore headaches for finals/etc
        pair_sorted = [tuple(sorted(pair)) for pair in zip(team_name, opp_name)]
        g["team_1"], g["team_2"] = zip(*pair_sorted)

        # Make a mask for whether the 'For' column is team1 or team2
        is_team_as_t1 = team_name.eq(g["team_1"])
        g["t1_points"] = g["For"].where(is_team_as_t1, g["Against"])
        g["t2_points"] = g["Against"].where(is_team_as_t1, g["For"])

        # Fix the scoring columns:
        scoring_df = g[["Scoring"]]
        scoring_df.columns = ["Scoring_For", "Scoring_Against"]
        qpoints_flat = match_utils.flatten_qcols(
            match_utils.scoring_points_per_quarter(scoring_df)
        )
        t12_scoring = match_utils.for_against_to_t1_t2(qpoints_flat, is_team_as_t1)
        g = g.join(t12_scoring)

        # Make our gameID column
        g["gameID"] = g.apply(
            lambda r: f"{r.Date.date()}-R-{r.Rnd}-{r.team_1}-{r.team_2}", axis=1
        )

        # Set the result- have either winning team name or 'draw'
        cond_win = g["t1_points"].gt(g["t2_points"])
        cond_loss = g["t1_points"].lt(g["t2_points"])

        g["result"] = np.select(
            [cond_win, cond_loss], [g["team_1"], g["team_2"]], default="draw"
        )
        g["margin"] = np.abs(g["t1_points"] - g["t2_points"])
        g["total_points"] = g["t1_points"] + g["t2_points"]

        games.append(
            g.drop(
                [
                    "Opponent",
                    "Result",
                    "Margin",
                    "Team",
                    "Scoring",
                    "For",
                    "Against",
                    "W-D-L",
                    "T",
                ],
                axis=1,
            )
        )

    match_df = pd.concat(games, axis=0)

    # Drop all rows where team_2 is 'university'
    match_df = match_df.loc[match_df.team_2 != "university"]

    match_df = match_df.set_index("gameID").sort_values("Date")
    match_df = match_df[~match_df.index.duplicated(keep="first")]

    # Set the team IDs
    match_df["team_1_id"] = match_df["team_1"].map(team_map)
    match_df["team_2_id"] = match_df["team_2"].map(team_map)
    match_df["team_1_id"] = match_df["team_1_id"].where(
        ~match_df["team_1_id"].isna(), match_df["team_1"].map(alias_map)
    )
    match_df["team_2_id"] = match_df["team_2_id"].where(
        ~match_df["team_2_id"].isna(), match_df["team_2"].map(alias_map)
    )

    for season in seasons:
        p = Path(f"data/stats-team-match/season={season}")
        p.mkdir(exist_ok=True, parents=True)
        match_df.loc[match_df.year == season].to_parquet(p / "stats-team-match.parquet")
