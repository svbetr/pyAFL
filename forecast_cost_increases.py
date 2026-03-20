import pandas as pd
import numpy as np

leg_df = pd.read_parquet("data/sgm_leg_data_results.parquet")

leg_df[["DI", "TK", "MK", "GL"]] = leg_df[["DI", "TK", "MK", "GL"]].fillna(0)

# See how many things (disposals, tackles, goals) a leg missed by
leg_df["is_disposal_leg"] = leg_df["stat"].eq("disposals")

# See how many things (disposals, tackles, goals) a leg missed by
leg_df["n_missed_by"] = np.nan
for stat_column, counts_column in zip(
    ["disposals", "tackles", "marks", "goals"], ["DI", "TK", "MK", "GL"]
):
    leg_df.loc[leg_df.stat == stat_column, "n_missed_by"] = (
        leg_df.loc[leg_df.stat == stat_column, "threshold"]
        - leg_df.loc[leg_df.stat == stat_column, counts_column]
    ).clip(0)

leg_df["disp_miss_exact"] = np.where(
    leg_df["is_disposal_leg"] & (~leg_df["leg_won"]),
    leg_df["n_missed_by"].astype("Int64"),
    pd.NA,
)

# Find all disposal legs
disp_only = leg_df.loc[
    leg_df["stat"].eq("disposals"), ["bet_id", "leg_won", "n_missed_by"]
].copy()
# Sort so losses (biggest first) appear before wins
disp_only = disp_only.sort_values(["bet_id", "n_missed_by"], ascending=[True, False])

# this makes an ordered list of how many disposals each leg lost by, grouped at the bet level
# It's a ragged list of
disp_all_list = (
    disp_only.groupby("bet_id")["n_missed_by"]
    .apply(list)
    .rename("disp_miss_by_including_wins")
)

# Make our overall bets table by combining the legs
bets = leg_df.groupby(["bet_id"]).agg(
    season=("year", "first"),
    event_date=("date", "first"),
    user_residence_state=("user_residence_state", "first"),
    event_country=("event_country", "first"),
    turnover_bonus_bet=("turnover_bonus_bet", "sum"),
    turnover_cash=("turnover_cash", "sum"),
    turnover_total=("turnover_total", "sum"),
    bet_fixed_odds=("bet_fixed_odds", "first"),
    payout=("payout", "sum"),
    net_win=("net_win", "sum"),
    gross_win=("gross_win_cash", "sum"),
    min_leg_price=("leg_price", "min"),
    n_legs=("sgm_leg_count", "max"),
    n_disp_legs=("is_disposal_leg", "sum"),
    n_winning_legs=("leg_won", "sum"),
)
bets["bet_won"] = bets["payout"] < 0
bets["lost_by_one_leg"] = bets["n_legs"] - bets["n_winning_legs"] == 1
bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(float)
bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]

lifeline_bets_N = bets.query(
    "season == 2025 and n_legs >= 4 and n_disp_legs >= 1 and bet_fixed_odds >= 3 and min_leg_price >= 1.2"
)["gross_win"].count()
lifeline_bets_turnover_cash = bets.query(
    "season == 2025 and n_legs >= 4 and n_disp_legs >= 1 and bet_fixed_odds >= 3 and min_leg_price >= 1.2"
)["turnover_cash"].sum()

all_SGM_bets_N = bets.query("season == 2025")["turnover_cash"].count()
all_SGM_bets_turnover_cash = bets.query("season == 2025")["turnover_cash"].sum()

df = pd.read_csv("2025_lifeline_data_multi_lifelines.csv")
N_bets = bets.loc[bets.n_legs >= 4].groupby("season")["event_country"].count()
N_bets.name = "N_SGMs_gt_4_legs"
df = df.merge(N_bets, left_on="season", right_index=True)
df["bet_save_fraction"] = df["worst_case_bets"] / df["N_SGMs_gt_4_legs"]

# Fit a simple linear regression to bet_save_fraction vs bet save fraction
p = np.poly1d(
    np.polyfit(
        df.loc[df.expected_payout > 0, "bet_save_fraction"] * 100,
        df.loc[df.expected_payout > 0, "expected_payout"] / 1e3,
        deg=1,
    )
)

extra_cost_per_1pc_saved = p[1]

print(
    f"Every extra 1% of SGMs saved with Wildcards costs us ${extra_cost_per_1pc_saved:.0f}k"
)


# Do the final costings

N_saved = (
    low_final.query("season == 2025")["worst_case_bets"].sum()
    + high_final.query("season == 2025")["worst_case_bets"].sum()
)
total_cost = (
    low_final.loc[2025, "expected_payout"].sum()
    + high_final.loc[2025, "expected_payout"].sum()
)

save_fraction = N_saved / lifeline_bets_N
lifeline_fraction = lifeline_bets_N / all_SGM_bets_N

lifeline_bets_turnover_cash_fraction = (
    lifeline_bets_turnover_cash / all_SGM_bets_turnover_cash
)

# in 2025, the lifeline bet turnover fraction was 0.19 (lifeline_bets_turnover_cash_fraction)
# predict it will go to 50% (i.e. a stretch)
forecast_lifeline_percentage_of_turnover = 0.5

# Cost as a fraction of turnover is roughly 17%
cost_as_a_fraction_of_turnover = 280_000 / lifeline_bets_turnover_cash


# MOdelled uplift for cash turnover in 2026 season
forecast_2026_season_turnover_cash_uplift = 1.5

total_forecast_cost = (
    forecast_lifeline_percentage_of_turnover
    * all_SGM_bets_turnover_cash
    * forecast_2026_season_turnover_cash_uplift
    * cost_as_a_fraction_of_turnover
)

print(
    f"The total forecasted cost for the product in 2026 is ${total_forecast_cost:,.2f}"
)
