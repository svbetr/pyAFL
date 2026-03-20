import pandas as pd
from pyAFL.utils import snowflake_utils, name_utils as nu, match_utils, leg_utils as lu
import duckdb as ddb  # type: ignore

tbs_sql = """
select
    B.bet_id,
    B.client_id,
    B.USER_RESIDENCE_STATE,
    B.event_country,
    CD.pre_gen_ngr_band,
    B.bet_leg_id,
    event_start_at::date as date,
    SGM_LEG_COUNT,
    TURNOVER_CASH,
    TURNOVER_BONUS_BET,
    TURNOVER_TOTAL,
    GROSS_WIN_CASH,
    GROSS_WIN_TOTAL,
    NET_WIN,
    B.PAYOUT,
    BET_FIXED_ODDS,
    MASTER_EVENT,
    EVENT_CLASS,
    EVENT,
    OUTCOME,
    BD.FIXEDPRICE as leg_price,
    (BD.PAYOUTPRICE > 0) as LEG_WON,
    (BD.payout > 0)  AS BET_WON
from dwh.core.vw_bets B
join tbs.dbo.betdetail BD
  on BD.betdetailid = B.bet_leg_id
join dwh.core.vw_client C
  on C.client_id = B.client_id
join dwh.core.tbl_crm_daily CD
  on CD.client_id = B.client_id
 and CD.date = B.event_start_at::date
where
    master_event_type = 'Sports'
    and event_type = 'Australian Rules'
    and MASTER_CATEGORY = 'AFL'
    and bet_type_code = 'SGMUL'
    and event_start_at::date >= '2022-01-01'
    and SGM_LEG_COUNT > 1
    and is_resulted
    and is_reportable
    and turnover_cash > 0
    and not C.client_profile_is_negative_factor
    and BD.mds_valid_to_dts is null;
"""

config = snowflake_utils.get_snowflake_connection_dict()
df = snowflake_utils.read_snowflake_query_to_df(config, tbs_sql).rename(
    str.lower, axis=1
)

# Fix some column types
df["bet_id"] = pd.to_numeric(df["bet_id"])
df["bet_leg_id"] = pd.to_numeric(df["bet_leg_id"])

# Add the home team and away team columns
pat = r"\s+v(?:s\.?)?\s+"  # ' v ', ' vs ', ' vs.' etc.
df[["homeTeam", "awayTeam"]] = df["master_event"].str.split(
    pat, n=1, regex=True, expand=True
)
team_name = df["homeTeam"].apply(match_utils.consistent_name)
opp_name = df["awayTeam"].apply(match_utils.consistent_name)

# Alphabetical order of teams- to ignore headaches for finals/etc
pair_sorted = [tuple(sorted(pair)) for pair in zip(team_name, opp_name)]
df["team_1"], df["team_2"] = zip(*pair_sorted)
df = df.drop(["homeTeam", "awayTeam"], axis=1)

df["date"] = pd.to_datetime(df["date"])
df["year"] = pd.to_datetime(df["date"]).dt.isocalendar()["year"]

# Fix a bad date for some legs in a Brisbane/Carlton match
df.loc[
    (df.date == "2025-07-09")
    & (df.team_1 == "brisbane-lions")
    & (df.team_2 == "carlton"),
    "date",
] = "2025-07-10"

# Join to the data we have
con = ddb.connect()

# Get the teams
teams = con.execute("SELECT * FROM parquet_scan('data/dim-team/*.parquet');").df()
# Add a draw team with an ID of 0
draw_row = pd.DataFrame(
    dict(teamID=0, teamName="Draw", displayName="Draw"), index=[max(teams.index) + 1]
)
teams = pd.concat([teams, draw_row])
teams["teamID"] = pd.to_numeric(teams["teamID"])

matches = con.execute("SELECT * FROM parquet_scan('data/dim-match/*.parquet');").df()
matches["Date"] = pd.to_datetime(matches["Date"].dt.date)
matches["team_1_id"] = matches["team_1_id"].astype(int)
matches["team_2_id"] = matches["team_2_id"].astype(int)

players = con.execute("SELECT * FROM parquet_scan('data/dim-player/*.parquet');").df()

# Add our alias table
try:
    aliases = pd.read_csv(
        "data/player_aliases/player_aliases.csv", dtype={"canonical_player_id": "Int64"}
    )
except FileNotFoundError:
    aliases = pd.DataFrame(
        columns=[
            "alias",
            "alias_key",
            "canonical_name",
            "canonical_player_id",
        ]
    )

# get our team aliases
try:
    team_aliases = pd.read_csv("data/dim-team/team_aliases.csv")
except FileNotFoundError:
    team_aliases = pd.DataFrame(columns=["alias", "teamID", "displayName"])

team_alias_dict = dict(zip(team_aliases["alias"], team_aliases["displayName"]))

# Make sure our teamIDs are correct before we merge

df["team_1"] = df["team_1"].replace(team_alias_dict)
df["team_2"] = df["team_2"].replace(team_alias_dict)

# # Add the team IDs
merged = pd.merge(
    df, teams[["teamName", "teamID"]], left_on="team_1", right_on="teamName", how="left"
)
merged = merged.rename(columns={"teamID": "team1_ID"}).drop(columns=["teamName"])

merged = pd.merge(
    merged, teams[["teamName", "teamID"]], left_on="team_2", right_on="teamName"
)
merged = merged.rename(columns={"teamID": "team2_ID"}).drop(columns=["teamName"])

# Add the Match ID
merged = pd.merge(
    merged,
    matches[["gameID", "team_1_id", "team_2_id", "Date"]],
    left_on=["team1_ID", "team2_ID", "date"],
    right_on=["team_1_id", "team_2_id", "Date"],
    how="left",
).drop(["Date", "team_1_id", "team_2_id"], axis=1)


# Player names
# Make a name_key column by:
# - Taking the outcome column value is the event is anytime goalscorer
# - Otherwise taking anything after a dash in the event column (e.g. "Total Goals - Jack Gunston")
# - Else returning NaN (i.e. if it's a team/margin/etc leg)
merged["name_key"] = (
    merged["outcome"]
    .where(merged["event"].eq("Anytime Goalscorer"))
    .fillna(merged["event"].str.extract(r"[-–—]\s*(.*)$", expand=False).str.strip())
)
merged["name_key"] = merged["name_key"].apply(nu.normalize_name)
final = (
    pd.merge(merged, players, left_on="name_key", right_on="name_key", how="left")
    .drop(columns=["firstName", "lastName"])
    .rename(columns=dict(name="playerName"))
)


# Now fix up some players with names that don't match
needs_alias = final["playerID"].isna()
if needs_alias.any():
    # Join catalog -> aliases on alias_key, then aliases -> players on canonical_player_id
    catalog_alias = final.loc[needs_alias].merge(
        aliases[["alias_key", "canonical_player_id", "canonical_name"]],
        left_on="name_key",
        right_on="alias_key",
        how="left",
    )
    catalog_alias["canonical_player_id"] = pd.to_numeric(
        catalog_alias["canonical_player_id"]
    ).astype(float)
    final["playerID"] = pd.to_numeric(final["playerID"])

    # Fill back into matched
    final.loc[needs_alias, ["playerID", "playerName"]] = catalog_alias[
        ["canonical_player_id", "canonical_name"]
    ].values


# Normalize the legs to make it easier to see when legs lose by a small number
leg_normalised = final.apply(lu.normalize_leg, axis=1, result_type="expand")
final = pd.concat([final, leg_normalised], axis=1)

# Get the player/match stats
player_match_stats = (
    con.execute(
        "select * from parquet_scan('data/stats-player-match/season=*/*.parquet');"
    )
    .df()
    .fillna(0.0)
)


# JOin to the final table
combined = final.merge(
    player_match_stats[
        [
            "playerID",
            "gameID",
            "KI",
            "MK",
            "HB",
            "DI",
            "GL",
            "BH",
            "HO",
            "TK",
            "RB",
            "IF",
            "CL",
            "CG",
            "FF",
            "FA",
            "BR",
            "CP",
            "UP",
            "CM",
            "MI",
        ]
    ],
    left_on=["playerID", "gameID"],
    right_on=["playerID", "gameID"],
    how="left",
)

# Add the match stats
# Get the player/match stats
match_stats = (
    con.execute(
        "select * from parquet_scan('data/stats-team-match/season=*/*.parquet');"
    )
    .df()
    .fillna(0.0)
)
# JOin to the final table
combined = combined.merge(
    match_stats[
        ["gameID", "total_points", "team_1", "t1_points", "team_2", "t2_points"]
    ],
    left_on=["gameID"],
    right_on=["gameID"],
    how="left",
)

# combined.to_parquet("data/sgm_leg_data_results.parquet")

# # Now only keep bets where we have all of their legs
# bets = evaluated_legs.groupby(["bet_id"]).agg(
#     all_legs=("sgm_leg_count", "max"), legs_in_table=("sgm_leg_count", "count")
# )
# bets["missing_legs"] = bets["all_legs"] - bets["legs_in_table"]
# fully_evaluated_bets = bets.loc[bets["missing_legs"] == 0].index


# fully_evaluted_SGMs = evaluated_legs.loc[
#     evaluated_legs.bet_id.isin(fully_evaluated_bets)
# ]

# print(f"We have {len(fully_evaluated_bets)} bets with {len(fully_evaluted_SGMs)} legs.")

# sgm_leg_filename = "data/SGM_legs.parquet"
# print(f"Saving to {sgm_leg_filename}")

# # remove any tuples in the fully_evaluated_SGM column
# mask = fully_evaluted_SGMs["actual"].map(type).eq(tuple)
# fully_evaluted_SGMs.loc[mask, "actual"] = -999
# fully_evaluted_SGMs.to_parquet(sgm_leg_filename)
