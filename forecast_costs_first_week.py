import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import pyAFL.utils.lifeline_utils as lu


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
    user_residence_state=("user_residence_state", "first"),
    event_country=("event_country", "first"),
    band=("pre_gen_ngr_band", "first"),
    event_date=("date", "first"),
    turnover_bonus_bet=("turnover_bonus_bet", "sum"),
    turnover_cash=("turnover_cash", "sum"),
    turnover_total=("turnover_total", "sum"),
    bet_fixed_odds=("bet_fixed_odds", "first"),
    payout=("payout", "sum"),
    gross_win=("gross_win_cash", "sum"),
    net_win=("net_win", "sum"),
    n_legs=("sgm_leg_count", "max"),
    n_disp_legs=("is_disposal_leg", "sum"),
    n_winning_legs=("leg_won", "sum"),
)
bets["bet_won"] = bets["payout"] < 0
bets["lost_by_one_leg"] = bets["n_legs"] - bets["n_winning_legs"] == 1
bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(float)
bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]

# Merge onto the list of how many misses each disposal leg had
bets = bets.merge(disp_all_list, how="left", left_index=True, right_index=True)

# Get our mask for if all losing legs are disposals
all_losing_are_disposals = (
    leg_df.loc[~leg_df["leg_won"]]
    .groupby("bet_id")["is_disposal_leg"]
    .all()
    .rename("all_losing_are_disposals")
)
bets = bets.merge(
    all_losing_are_disposals, left_index=True, right_index=True, how="left"
)
# Bets that win are NaN here- fill with False
bets["all_losing_are_disposals"] = bets["all_losing_are_disposals"].fillna(False)

# Add total fees
rpf_taxes = pd.read_parquet("data/fees/rpf_costs.parquet")
poc_taxes = pd.read_parquet("data/fees/poc_costs.parquet")

bets = bets.merge(
    poc_taxes[["state", "poc_rate", "comp_gross_win"]],
    left_on="user_residence_state",
    right_on="state",
).drop("state", axis=1)
bets["product_fee_rate"] = rpf_taxes["product_fee_rate"].values[0]

# add the fees
bets["rpf_fee"] = bets["turnover_cash"] * bets["product_fee_rate"]
bets["poc_fee"] = bets["turnover_cash"] * bets["poc_rate"] * bets["comp_gross_win"]

bets["week_of_year"] = bets.event_date.dt.isocalendar().week

low = bets.loc[bets.band.isin(["NGR Band 7: $0 - $25", "NGR Band 6: $25 - $50"])]
high = bets.loc[
    bets.band.isin(
        [
            "NGR Band 5: $50 - $100",
            "NGR Band 4: $100 - $250",
            "NGR Band 3: $250 - $500",
            "NGR Band 2: $500 - $1000",
            "NGR Band 1: $1000+",
            "VIP Program",
        ]
    )
]

# lifeline_product = pd.DataFrame(
#     data=dict(
#         tier=["low"] * 9 + ["high"] * 9,
#         legs=[4, 5, 6, 7, 8, 9, 10, 11, 12] + [4, 5, 6, 7, 8, 9, 10, 11, 12],
#         lifelines=[
#             [1],
#             [1],
#             [2, 2],
#             [2, 2],
#             [2, 2, 2],
#             [2, 2, 2],
#             [2, 2, 2, 2],
#             [2, 2, 2, 2],
#             [2, 2, 2, 2],
#         ]
#         + [
#             [2],
#             [2],
#             [4, 4],
#             [4, 4],
#             [3, 3, 3],
#             [3, 3, 3],
#             [4, 4, 4, 4],
#             [4, 4, 4, 4],
#             [4, 4, 4, 4],
#         ],
#     )
# )


lifeline_product = pd.DataFrame(
    data=dict(
        tier=["low"] * 9 + ["high"] * 9,
        legs=[4, 5, 6, 7, 8, 9, 10, 11, 12] + [4, 5, 6, 7, 8, 9, 10, 11, 12],
        lifelines=[
            [2],
            [2, 2],
            [2, 2, 3],
            [2, 2, 3],
            [2, 3, 4],
            [2, 3, 4],
            [3, 3, 3, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ]
        + [
            [4],
            [4, 4],
            [3, 4, 4],
            [3, 4, 4],
            [4, 4, 4],
            [4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
            [4, 4, 4, 4],
        ],
    )
)

# Merge together to get bets with lifelines
low_with_lifelines = low.merge(
    lifeline_product.loc[lifeline_product.tier == "low"],
    left_on="n_legs",
    right_on="legs",
)
high_with_lifelines = high.merge(
    lifeline_product.loc[lifeline_product.tier == "high"],
    left_on="n_legs",
    right_on="legs",
)
bets_with_lifelines = pd.concat([low_with_lifelines, high_with_lifelines], axis=0)
bets_with_lifelines["week_of_year"] = (
    bets_with_lifelines["event_date"].dt.isocalendar().week
)

bets_with_lifelines["bet_would_win_with_lifelines"] = bets_with_lifelines.apply(
    lambda row: lu.would_win_with_lifelines(
        row["disp_miss_by_including_wins"], lifelines=row["lifelines"]
    ),
    axis=1,
)

# Get the expected liability
bets_with_lifelines["expected_liability"] = bets_with_lifelines.apply(
    lambda row: lu._expected_liability_row(row, lifelines=row["lifelines"]),
    axis=1,
)
# Worst case payout after lifelines
bets_with_lifelines["worst_case_payout_after_lifelines"] = (
    bets_with_lifelines["liability"]
    * bets_with_lifelines["bet_would_win_with_lifelines"]
    * (
        1 - bets_with_lifelines["bet_won"]
    )  # Only count the payout if lifelines saved the bet
    * (
        bets_with_lifelines["all_losing_are_disposals"]
    )  # multiply by zero if not the case that all losing are disposals
)

# Expected payput after lifelines
bets_with_lifelines["expected_payout_after_lifelines"] = (
    bets_with_lifelines["expected_liability"]
    * bets_with_lifelines["bet_would_win_with_lifelines"]
    * (
        1 - bets_with_lifelines["bet_won"]
    )  # Only count the payout if lifelines saved the bet
    * (
        bets_with_lifelines["all_losing_are_disposals"]
    )  # multiply by zero if not the case that all losing are disposals
)


# Find the average cost per week of the season
mean_expected_cost = (
    bets_with_lifelines.groupby(["season", "week_of_year"])[
        "expected_payout_after_lifelines"
    ]
    .sum()
    .loc[2025]
).mean()
std_expected_cost = (
    bets_with_lifelines.groupby(["season", "week_of_year"])[
        "expected_payout_after_lifelines"
    ]
    .sum()
    .loc[2025]
).std()


mean_worst_case_cost = (
    bets_with_lifelines.groupby(["season", "week_of_year"])[
        "worst_case_payout_after_lifelines"
    ]
    .sum()
    .loc[2025]
).mean()
std_worst_case_cost = (
    bets_with_lifelines.groupby(["season", "week_of_year"])[
        "worst_case_payout_after_lifelines"
    ]
    .sum()
    .loc[2025]
).std()
