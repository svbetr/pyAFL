import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
from pyAFL.utils.lifeline_utils import lifeline_summary

if __name__ == "__main__":

    leg_df_master = pd.read_parquet("data/sgm_leg_data_results.parquet")

    bet_fixed_odds_constraints = [0, 3, 3.5, 4, 4.5, 5, 5.5]
    leg_fixed_odds_constraints = [0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

    results = []

    for b_constraint in bet_fixed_odds_constraints:
        for l_constraint in leg_fixed_odds_constraints:

            print(f"Bet Fixed Odds Minimum: {b_constraint}")
            print(f"Leg Fixed Odds Minimum: {l_constraint}")

            # Apply the bet fixed odds constraint at the top
            # The leg fixed odds constraint comes later
            mask = leg_df_master.bet_fixed_odds >= b_constraint

            leg_df = leg_df_master.loc[mask].copy()

            leg_df[["DI", "TK", "MK", "GL"]] = leg_df[["DI", "TK", "MK", "GL"]].fillna(
                0
            )

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
                user_residence_state=("user_residence_state", "first"),
                event_country=("event_country", "first"),
                band=("pre_gen_ngr_band", "first"),
                event_date=("date", "first"),
                turnover_bonus_bet=("turnover_bonus_bet", "sum"),
                turnover_cash=("turnover_cash", "sum"),
                turnover_total=("turnover_total", "sum"),
                bet_fixed_odds=("bet_fixed_odds", "first"),
                min_leg_price=("leg_price", "min"),
                payout=("payout", "sum"),
                gross_win=("gross_win_cash", "sum"),
                net_win=("net_win", "sum"),
                n_legs=("sgm_leg_count", "max"),
                n_disp_legs=("is_disposal_leg", "sum"),
                n_winning_legs=("leg_won", "sum"),
            )

            # Apply the leg fixed odds constraint
            bets = bets.loc[bets.min_leg_price >= l_constraint].copy()

            bets["bet_won"] = bets["payout"] < 0
            bets["lost_by_one_leg"] = bets["n_legs"] - bets["n_winning_legs"] == 1
            bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(
                float
            )
            bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]

            # Merge onto the list of how many misses each disposal leg had
            bets = bets.merge(
                disp_all_list, how="left", left_index=True, right_index=True
            )

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
            bets["all_losing_are_disposals"] = bets["all_losing_are_disposals"].fillna(
                False
            )

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
            bets["poc_fee"] = (
                bets["turnover_cash"] * bets["poc_rate"] * bets["comp_gross_win"]
            )

            low = bets.loc[
                bets.band.isin(["NGR Band 7: $0 - $25", "NGR Band 6: $25 - $50"])
            ]
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

            lifeline_product = pd.DataFrame(
                data=dict(
                    tier=["low"] * 9 + ["high"] * 9,
                    legs=[4, 5, 6, 7, 8, 9, 10, 11, 12]
                    + [4, 5, 6, 7, 8, 9, 10, 11, 12],
                    lifelines=[
                        [1],
                        [1],
                        [2, 2],
                        [2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2],
                    ]
                    + [
                        [2],
                        [2],
                        [4, 4],
                        [4, 4],
                        [3, 3, 3],
                        [3, 3, 3],
                        [4, 4, 4, 4],
                        [4, 4, 4, 4],
                        [4, 4, 4, 4],
                    ],
                )
            )

            low_tier_results = []
            for row in tqdm(
                lifeline_product.loc[lifeline_product.tier == "low"].itertuples(
                    index=False
                )
            ):
                tmp = lifeline_summary(low, n_legs=row.legs, lifelines=row.lifelines)
                tmp.insert(0, "n_legs", row.legs)
                tmp.insert(1, "lifelines", ",".join([str(l) for l in row.lifelines]))
                tmp["gross_win_all_lifeline_eligible_bets"] = low.loc[
                    (low.season == 2025)
                    & (low.n_legs == row.legs)
                    & (low.n_disp_legs >= len(row.lifelines)),
                    "gross_win",
                ].sum()
                low_tier_results.append(tmp)
            low_final = pd.concat(low_tier_results, axis=0)

            high_tier_results = []
            for row in tqdm(
                lifeline_product.loc[lifeline_product.tier == "high"].itertuples(
                    index=False
                )
            ):
                tmp = lifeline_summary(high, n_legs=row.legs, lifelines=row.lifelines)
                tmp.insert(0, "n_legs", row.legs)
                tmp.insert(1, "lifelines", ",".join([str(l) for l in row.lifelines]))
                tmp["gross_win_all_lifeline_eligible_bets"] = high.loc[
                    (high.season == 2025)
                    & (high.n_legs == row.legs)
                    & (high.n_disp_legs >= len(row.lifelines)),
                    "gross_win",
                ].sum()
                high_tier_results.append(tmp)
            high_final = pd.concat(high_tier_results, axis=0)

            total_cost = (
                low_final.loc[2025, "expected_payout"].sum()
                + high_final.loc[2025, "expected_payout"].sum()
            ) / 1e3
            gross_win_total = (
                low.loc[low.season == 2025, "gross_win"].sum()
                + high.loc[high.season == 2025, "gross_win"].sum()
            ) / 1e6
            fees_saved = (
                low_final.loc[2025, "poc_fees"].sum()
                + high_final.loc[2025, "poc_fees"].sum()
            ) / 1e3

            print(f"\tLifeline product is:\n{lifeline_product}\n")
            print(f"\tTotal cost: ${total_cost:,.2f}k")
            print(f"\tFees Saved: ${fees_saved:,.2f}k")
            print(f"\tTotal Gross Win: ${gross_win_total:,.2f}M")
            print(
                f"\tCost as a %age of Gross Win: {total_cost / (gross_win_total * 1e3):.1%}"
            )

            # Analyse the lifeline gross win differences per cohort
            low_tier_gross_win_delta = low_final.loc[
                2025, "gross_win_of_original_bets"
            ].sum()
            low_tier_gross_win_after_lifelines = (
                low.loc[low.season == 2025, "gross_win"].sum()
                - low_tier_gross_win_delta
            )

            high_tier_gross_win_delta = high_final.loc[
                2025, "gross_win_of_original_bets"
            ].sum()
            high_tier_gross_win_after_lifelines = (
                high.loc[high.season == 2025, "gross_win"].sum()
                - high_tier_gross_win_delta
            )

            print("Gross win of bets not saved by lifelines:")
            print(f"Bands 6-7: ${low_tier_gross_win_after_lifelines / 1e6:.2f}M")
            print(f"Bands 1-5: ${high_tier_gross_win_after_lifelines / 1e6:.2f}M")

            # Come up with the cost per number of legs:

            low_summary = low_final.loc[
                2025,
                [
                    "n_legs",
                    "lifelines",
                    "expected_payout",
                    "gross_win_of_original_bets",
                    "gross_win_all_lifeline_eligible_bets",
                    "worst_case_bets",
                ],
            ]
            low_summary["gross_win_losing_bets_post_lifelines"] = (
                low_summary["gross_win_all_lifeline_eligible_bets"]
                - low_summary["gross_win_of_original_bets"]
            )
            low_summary["expected_payout_frac"] = (
                low_summary["expected_payout"]
                / low_summary["gross_win_losing_bets_post_lifelines"]
            )

            high_summary = high_final.loc[
                2025,
                [
                    "n_legs",
                    "lifelines",
                    "expected_payout",
                    "gross_win_of_original_bets",
                    "gross_win_all_lifeline_eligible_bets",
                    "worst_case_bets",
                ],
            ]
            high_summary["gross_win_losing_bets_post_lifelines"] = (
                high_summary["gross_win_all_lifeline_eligible_bets"]
                - high_summary["gross_win_of_original_bets"]
            )
            high_summary["expected_payout_frac"] = (
                high_summary["expected_payout"]
                / high_summary["gross_win_losing_bets_post_lifelines"]
            )

            # low_summary.to_csv("summary_results/low_bands_summary.csv")
            # high_summary.to_csv("summary_results/high_bands_summary.csv")

            constrained_summary = pd.concat(
                [high_summary.sum(), low_summary.sum()], axis=1
            ).T.drop(columns=["lifelines", "expected_payout_frac", "n_legs"])
            constrained_summary.index = ["low", "high"]
            constrained_summary["bet_constraint"] = b_constraint
            constrained_summary["leg_constraint"] = l_constraint
            constrained_summary = constrained_summary.reset_index()

            results.append(constrained_summary)

    overall_summary = (
        pd.concat(results, axis=0)
        .set_index(["bet_constraint", "leg_constraint", "index"])
        .groupby(level=[0, 1])
        .sum()
    )
