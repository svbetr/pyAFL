import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from pyAFL.utils.lifeline_utils import lifeline_summary

if __name__ == "__main__":

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
    disp_only = disp_only.sort_values(
        ["bet_id", "n_missed_by"], ascending=[True, False]
    )

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

    lifeline_numbers = [1, 2, 3, 4, 5]
    combos = [
        list(reversed(c))
        for r in range(1, 6)
        for c in itertools.combinations_with_replacement(lifeline_numbers, r)
    ]

    results = []
    for n_legs in tqdm([4, 5, 6, 7, 8, 9, 10, 11, 12]):
        for lifeline in tqdm(combos):
            tmp = lifeline_summary(bets, n_legs=n_legs, lifelines=lifeline)
            tmp.insert(0, "n_legs", n_legs)
            tmp.insert(1, "lifelines", ",".join([str(l) for l in lifeline]))
            results.append(tmp)
    final = pd.concat(results, axis=0)
    final.loc[2025].to_csv("2025_lifeline_data_multi_lifelines.csv")
