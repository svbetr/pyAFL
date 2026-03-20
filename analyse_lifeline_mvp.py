import pandas as pd
import numpy as np
from tqdm import tqdm
from pyAFL.utils.lifeline_utils import lifeline_summary

# --- Constants ---

WILDCARD_LEG_TYPES = ["disposals"]

LOW_BANDS = ["NGR Band 7: $0 - $25", "NGR Band 6: $25 - $50"]
HIGH_BANDS = [
    "NGR Band 5: $50 - $100",
    "NGR Band 4: $100 - $250",
    "NGR Band 3: $250 - $500",
    "NGR Band 2: $500 - $1000",
    "NGR Band 1: $1000+",
    "VIP Program",
]

STAT_TO_COLUMN = {
    "disposals": "DI",
    "tackles": "TK",
    "marks": "MK",
    "goals": "GL",
    "total_points": "total_points",
}

SUMMARY_COLS = [
    "n_legs",
    "lifelines",
    "expected_payout",
    "worst_case_bets",
    "worst_case_payout",
    "gross_win_of_original_bets",
    "gross_win_all_lifeline_eligible_bets",
]


# --- Data loading ---


def load_legs() -> pd.DataFrame:
    leg_df = pd.read_parquet("data/sgm_leg_data_results.parquet")
    print(f"We have {len(leg_df):,} legs from {leg_df.bet_id.nunique():,} bets")
    return leg_df


def load_fees() -> tuple[pd.DataFrame, pd.DataFrame]:
    rpf_taxes = pd.read_parquet("data/fees/rpf_costs.parquet")
    poc_taxes = pd.read_parquet("data/fees/poc_costs.parquet")
    return rpf_taxes, poc_taxes


# --- Feature engineering ---


def engineer_leg_features(leg_df: pd.DataFrame) -> pd.DataFrame:
    leg_df = leg_df.copy()
    leg_df[["DI", "TK", "MK", "GL"]] = leg_df[["DI", "TK", "MK", "GL"]].fillna(0)

    leg_df["is_wildcard_leg"] = leg_df["stat"].isin(WILDCARD_LEG_TYPES)

    print(
        f"We have {leg_df.is_wildcard_leg.sum()} wildcard legs"
        f" ({leg_df.is_wildcard_leg.sum() / len(leg_df):.1%})"
    )

    # Find how many each column missed by
    over_under_factor = np.where(leg_df["comparator"].isin([">", ">="]), 1, -1)
    leg_df["n_missed_by"] = np.nan
    for stat_column, counts_column in STAT_TO_COLUMN.items():
        mask = leg_df.stat == stat_column
        leg_df.loc[mask, "n_missed_by"] = (
            (leg_df.loc[mask, "threshold"] - leg_df.loc[mask, counts_column])
            * over_under_factor[mask]
        ).clip(0)

    return leg_df


def build_bets(leg_df: pd.DataFrame) -> pd.DataFrame:
    # Ordered list of how many each wildcard leg missed by, grouped at bet level
    wildcard_miss_by = (
        leg_df.loc[leg_df["stat"].isin(WILDCARD_LEG_TYPES), ["bet_id", "n_missed_by"]]
        .sort_values(["bet_id", "n_missed_by"], ascending=[True, False])
        .groupby("bet_id")["n_missed_by"]
        .apply(list)
        .rename("wildcard_miss_by_including_wins")
    )

    # Flag: all losing legs are wildcard legs (NaN for winning bets → fill False)
    all_losing_are_wildcard = (
        leg_df.loc[~leg_df["leg_won"]]
        .groupby("bet_id")["is_wildcard_leg"]
        .all()
        .rename("all_losing_are_wildcard")
    )

    bets = leg_df.groupby("bet_id").agg(
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
        n_disp_legs=("is_wildcard_leg", "sum"),
        n_winning_legs=("leg_won", "sum"),
    )
    bets["bet_won"] = bets["payout"] < 0
    bets["lost_by_one_leg"] = bets["n_legs"] - bets["n_winning_legs"] == 1
    bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(float)
    bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]

    bets = bets.merge(wildcard_miss_by, how="left", left_index=True, right_index=True)
    bets = bets.merge(
        all_losing_are_wildcard, how="left", left_index=True, right_index=True
    )
    bets["all_losing_are_wildcard"] = bets["all_losing_are_wildcard"].fillna(False)

    return bets


def add_fees(
    bets: pd.DataFrame, rpf_taxes: pd.DataFrame, poc_taxes: pd.DataFrame
) -> pd.DataFrame:
    bets = bets.merge(
        poc_taxes[["state", "poc_rate", "comp_gross_win"]],
        left_on="user_residence_state",
        right_on="state",
    ).drop("state", axis=1)
    bets["product_fee_rate"] = rpf_taxes["product_fee_rate"].values[0]
    bets["rpf_fee"] = bets["turnover_cash"] * bets["product_fee_rate"]
    bets["poc_fee"] = bets["turnover_cash"] * bets["poc_rate"] * bets["comp_gross_win"]
    bets["week_of_year"] = bets.event_date.dt.isocalendar().week

    return bets


# --- Lifeline product definition ---


def define_lifeline_product() -> pd.DataFrame:
    legs = [4, 5, 6, 7, 8, 9, 10, 11, 12]
    return pd.DataFrame(
        data=dict(
            tier=["low"] * 9 + ["high"] * 9,
            legs=legs + legs,
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


# --- Analysis ---


def run_tier_analysis(
    bets_tier: pd.DataFrame,
    lifeline_product: pd.DataFrame,
    tier_name: str,
    season: int = 2025,
) -> pd.DataFrame:

    # Cost up each lifeline
    results = []
    for row in tqdm(
        lifeline_product.loc[lifeline_product.tier == tier_name].itertuples(
            index=False
        ),
        desc=tier_name,
    ):
        tmp = lifeline_summary(bets_tier, n_legs=row.legs, lifelines=row.lifelines)
        tmp.insert(0, "n_legs", row.legs)
        tmp.insert(1, "lifelines", ",".join(str(l) for l in row.lifelines))
        tmp["gross_win_all_lifeline_eligible_bets"] = bets_tier.loc[
            (bets_tier.season == season)
            & (bets_tier.n_legs == row.legs)
            & (bets_tier.n_disp_legs >= len(row.lifelines)),
            "gross_win",
        ].sum()
        results.append(tmp)
    return pd.concat(results, axis=0)


def build_tier_summary(final: pd.DataFrame, season: int = 2025) -> pd.DataFrame:
    summary = final.loc[season, SUMMARY_COLS].copy()
    summary["gross_win_losing_bets_post_lifelines"] = (
        summary["gross_win_all_lifeline_eligible_bets"]
        - summary["gross_win_of_original_bets"]
    )
    summary["expected_payout_frac"] = (
        summary["expected_payout"] / summary["gross_win_losing_bets_post_lifelines"]
    )
    return summary


def print_summary(
    low_final: pd.DataFrame,
    high_final: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    lifeline_product: pd.DataFrame,
    season: int = 2025,
) -> None:

    # Calculate a few useful numbers
    total_cost = (
        low_final.loc[season, "expected_payout"].sum()
        + high_final.loc[season, "expected_payout"].sum()
    ) / 1e3
    worst_case_payout = (
        low_final.loc[season, "worst_case_payout"].sum()
        + high_final.loc[season, "worst_case_payout"].sum()
    ) / 1e6
    gross_win_total = (
        low.loc[low.season == season, "gross_win"].sum()
        + high.loc[high.season == season, "gross_win"].sum()
    ) / 1e6
    fees_saved = (
        low_final.loc[season, "poc_fees"].sum()
        + high_final.loc[season, "poc_fees"].sum()
    ) / 1e3

    print(f"Lifeline product is:\n{lifeline_product}\n")
    print(f"Total cost: ${total_cost:,.2f}k")
    print(f"Worse case cost: ${worst_case_payout:,.2f}M")
    print(f"Fees Saved: ${fees_saved:,.2f}k")
    print(f"Total Gross Win: ${gross_win_total:,.2f}M")
    print(f"Cost as a %age of Gross Win: {total_cost / (gross_win_total * 1e3):.1%}")

    low_gw_delta = low_final.loc[season, "gross_win_of_original_bets"].sum()
    high_gw_delta = high_final.loc[season, "gross_win_of_original_bets"].sum()
    print("Gross win of bets not saved by lifelines:")
    print(
        f"  Bands 6-7: ${(low.loc[low.season == season, 'gross_win'].sum() - low_gw_delta) / 1e6:.2f}M"
    )
    print(
        f"  Bands 1-5: ${(high.loc[high.season == season, 'gross_win'].sum() - high_gw_delta) / 1e6:.2f}M"
    )


# --- Entry point ---


if __name__ == "__main__":

    # Work through the stages
    # Get our data and add things like 'n_missed_by'.
    leg_df = load_legs()
    leg_df = engineer_leg_features(leg_df)

    rpf_taxes, poc_taxes = load_fees()
    bets = build_bets(leg_df)
    bets = add_fees(bets, rpf_taxes, poc_taxes)

    low = bets.loc[bets.band.isin(LOW_BANDS)]
    high = bets.loc[bets.band.isin(HIGH_BANDS)]

    lifeline_product = define_lifeline_product()
    low_final = run_tier_analysis(low, lifeline_product, "low")
    high_final = run_tier_analysis(high, lifeline_product, "high")

    print_summary(low_final, high_final, low, high, lifeline_product)

    low_summary = build_tier_summary(low_final)
    high_summary = build_tier_summary(high_final)

    low_summary.to_csv("summary_results/low_bands_summary.csv")
    high_summary.to_csv("summary_results/high_bands_summary.csv")
