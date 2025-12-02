import pandas as pd
import numpy as np
from tqdm import tqdm
from pyAFL.utils.lifeline_utils import lifeline_summary
from typing import Callable

LifelineStrategy = Callable[[pd.DataFrame], int]  # takes group, returns index


def bet_saved_by_lifeline(group):

    # Was each leg either a winner or saved by a lifeline?
    return (~group["leg_won"].all()) & (
        group["leg_won"] | group["lifeline_saves_leg"]
    ).all()


def build_bets(leg_df: pd.DataFrame) -> pd.DataFrame:
    g = leg_df.groupby("bet_id")

    bets = g.agg(
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
        n_winning_legs=("leg_won", "sum"),
    )

    bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]
    bets["lost_by_one_leg"] = bets["n_losing_legs"] == 1
    bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(float)

    bets["bet_won"] = bets["payout"] < 0

    bets["band_tier"] = np.nan
    bets.loc[
        bets.band.isin(["NGR Band 7: $0 - $25", "NGR Band 6: $25 - $50"]), "band_tier"
    ] = "low"
    bets.loc[
        bets.band.isin(
            [
                "NGR Band 5: $50 - $100",
                "NGR Band 4: $100 - $250",
                "NGR Band 3: $250 - $500",
                "NGR Band 2: $500 - $1000",
                "NGR Band 1: $1000+",
                "VIP Program",
            ],
        ),
        "band_tier",
    ] = "high"

    saved = g.apply(bet_saved_by_lifeline).rename("bet_saved_by_lifeline")
    bets = bets.join(saved)

    return bets


def run_lifeline_scenario(
    leg_df: pd.DataFrame, strategy: LifelineStrategy
) -> pd.DataFrame:
    df = leg_df.copy()

    # start with no lifelines
    df["lifeline_saves_leg"] = False

    # for each bet_id, choose one leg index according to the strategy
    chosen_idx = df.groupby("bet_id").apply(strategy)

    # drop groups where the strategy returned NA / None
    chosen_idx = chosen_idx.dropna()
    chosen_idx = chosen_idx.astype(int)

    # mark those legs as saved by lifeline
    df.loc[chosen_idx, "lifeline_saves_leg"] = True

    # now build the bets table (costs, summary, etc.)
    bets = build_bets(df)
    return bets


def summarise_results(df):

    # Add some columns for the entire cohort
    overall_results = df.groupby(["season", "band_tier"]).agg(
        n_bets=("liability", "count"),
        gross_win_all_lifeline_eligible_bets=("gross_win", "sum"),
    )

    # Add some columns for just the bets which lifelines save
    saved_results = (
        df.loc[df.bet_saved_by_lifeline]
        .groupby(["season", "band_tier"])
        .agg(
            expected_payout=("liability", "sum"),
            gross_win_saved_bets=("gross_win", "sum"),
        )
    )

    # join
    results = pd.concat([overall_results, saved_results], axis=1)

    # Add combined columns
    results["gross_win_losing_bets_post_lifelines"] = (
        results["gross_win_all_lifeline_eligible_bets"]
        - results["gross_win_saved_bets"]
    )
    results["payout_as_frac_of_gross"] = (
        results["expected_payout"] / results["gross_win_losing_bets_post_lifelines"]
    )
    return results.loc[2025]


def shortest_price_strategy(group: pd.DataFrame) -> int:
    if len(group) < 4:
        return pd.NA
    return group["leg_price"].idxmin()


def longest_price_strategy(group: pd.DataFrame) -> int:

    if len(group) < 4:
        return pd.NA
    return group["leg_price"].idxmax()


def random_leg_strategy(group: pd.DataFrame) -> int:

    if len(group) < 4:
        return pd.NA
    return np.random.choice(group.index)


def random_leg_strategy_weight_by_inverse_price(group: pd.DataFrame) -> int:

    if len(group) < 4:
        return pd.NA
    if np.any(group.leg_price < 1):
        return pd.NA

    probs = 1.0 / group.leg_price
    probs = probs / probs.sum()
    return np.random.choice(group.index, p=probs)


if __name__ == "__main__":

    leg_df = pd.read_parquet("data/sgm_leg_data_results.parquet")

    # Try different lifeline implementations
    strategies = {
        "shortest_price": shortest_price_strategy,
        "longest_price": longest_price_strategy,
        "random_leg": random_leg_strategy,
        "random_leg_weight_by_inverse_price": random_leg_strategy_weight_by_inverse_price,
    }

    results = []
    for name, strat in strategies.items():
        print(f"Running scenario: {name}")
        bets = run_lifeline_scenario(leg_df, strat)
        # if you have a helper to summarise cost:
        # summary = lifeline_summary(bets)
        # results[name] = summary
        summary = summarise_results(bets)  # or store bets if you prefer
        summary["strategy"] = name
        summary = summary.set_index("strategy", append=True)
        results.append(summary)
    results_df = pd.concat(results, axis=0)
