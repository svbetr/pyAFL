from pyAFL.players.models import Player
from pyAFL.teams import CURRENT_TEAMS
import pandas as pd
import re


def consistent_name(s: str) -> str:

    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "-", s)

    return re.sub(r"-+", "-", s).strip("-")


if __name__ == "__main__":

    print("Getting data for current teams")

    team_df = pd.read_csv("data/dim-team/teams.csv")
    alias_df = pd.read_csv("data/dim-team/team_aliases.csv")

    team_map = dict(zip(team_df["teamName"], team_df["teamID"]))
    alias_map = dict(zip(alias_df["alias"], alias_df["teamID"]))

    games = []
    for team in CURRENT_TEAMS:
        print(team)
        g = team.games
        g["Team"] = team.name
        g["Date"] = pd.to_datetime(
            g["Date"], format="%a %d-%b-%Y %I:%M %p", errors="raise"
        )

        team_name = g["Team"].apply(consistent_name)
        opp_name = g["Opponent"].apply(consistent_name)

        # Alphabetical order of teams- to ignore headaches for finals/etc
        pair_sorted = [tuple(sorted(pair)) for pair in zip(team_name, opp_name)]
        g["team_1"], g["team_2"] = zip(*pair_sorted)

        is_team_as_t1 = team_name.eq(g["team_1"])
        g["t1_points"] = g["For"].where(is_team_as_t1, g["Against"])
        g["t2_points"] = g["Against"].where(is_team_as_t1, g["For"])

        g["gameID"] = g.apply(
            lambda r: f"{r.Date.date()}-{r.team_1}-{r.team_2}", axis=1
        )

        g["total_points"] = g["t1_points"] + g["t2_points"]

        games.append(g)

    match_df = pd.concat(games, axis=0)

    # Drop 'University' games
    match_df = match_df.loc[match_df.Opponent != "University"]
    match_df = match_df.set_index("gameID")
    match_df[~match_df.index.duplicated(keep="first")]

    match_df["team_1_id"] = match_df["team_1"].map(team_map)
    match_df["team_2_id"] = match_df["team_2"].map(team_map)
    # match_df["team_1_id"] = match_df["team_1"].where(
    #     match_df["team_1"].isna(), match_df["team_1"].map(alias_map)
    # )
    # match_df["team_2_id"] = match_df["team_2"].where(
    #     match_df["team_2"].isna(), match_df["team_2"].map(alias_map)
    # )
