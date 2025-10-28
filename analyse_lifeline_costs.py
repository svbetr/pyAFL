import pandas as pd
import numpy as np
from math import comb
from functools import reduce
import operator as op
import itertools

MAX_BUCKET = 10  # Largest number of disposals we could save


def _expected_liability_row(row, n_lifelines: int):
    # n_losing_legs = total legs - winning legs
    n_losing = row.n_legs - row.n_winning_legs

    if n_lifelines >= row.n_disp_legs:
        return row["liability"]

    return (
        row["liability"]
        * comb(row.n_disp_legs - n_losing, n_lifelines - n_losing)
        / comb(row.n_disp_legs, n_lifelines)
    )


def make_disp_miss_list(s: pd.Series) -> tuple[int, ...]:
    # s is the disp_miss_exact column for a single bet
    v = s.dropna().astype(int).tolist()
    return tuple(sorted(v, reverse=True))  # largest miss first helps later


def make_disp_miss_hist(s: pd.Series) -> list[int]:
    v = s.dropna().astype(int)
    if v.empty:
        return [0] * MAX_BUCKET  # indices 0..9 represent misses 1..10+ for example
    clipped = np.clip(v, 1, MAX_BUCKET)
    # bincount expects 0-based; shift by -1
    hist = np.bincount(clipped - 1, minlength=MAX_BUCKET)
    return hist.tolist()


def success_probability(misses, lifelines):
    # misses: e.g. [5,3,2,0,0]
    # lifelines: e.g. [5,3,2]  (treated as distinct; order matters)
    legs = list(range(len(misses)))
    losing = {i for i, m in enumerate(misses) if m > 0}

    total = 0
    success = 0

    for chosen_legs in itertools.permutations(legs, r=len(lifelines)):
        total += 1

        # map chosen leg -> lifeline value
        assignment = {leg_idx: L for L, leg_idx in zip(lifelines, chosen_legs)}

        # 1) every losing leg must be targeted by some lifeline
        if not losing.issubset(assignment.keys()):
            continue

        # 2) and that lifeline must be strong enough
        if all(assignment[i] >= misses[i] for i in losing):
            success += 1

    return success / total if total else 0.0


def lifeline_summary(
    bets,
    n_legs: int,
    n_lifelines: int,
    season_col: str = "season",
    disposal_missed_by_column="disp_leg_missed_by_one",
):
    """
    Filters bets for an L-leg, K-lifeline product where the leg missed by
    1..K legs, computes expected liability, and aggregates by season.

    Note you can change whether you want the lifeline to be +1, +2, etc by changing the 'disposal missed by' column name.
    """
    # Base mask: correct product and the bet lost
    base = (bets.n_legs.eq(n_legs)) & (~bets.bet_won)

    # Accept any "missed by m" with m in 1..K (and optionally enforce wins == L - m)
    # This loops over the cases where bets miss by 1 for 1 lifeline, 1 or 2 for 2 lifelines, etc
    conds = []
    for m in range(1, n_lifelines + 1):
        c = bets[disposal_missed_by_column].eq(m) & bets.n_winning_legs.eq(n_legs - m)
        conds.append(c)

    # mask is winning bets and any of our conditions (miss by 1 for 1 lifeline, 1 or 2 for 2 lifelines, etc)
    mask = base & reduce(op.or_, conds)

    df = bets.loc[mask].copy()

    # Expected liability per row (pass the extra arg via apply kwargs)
    df["expected_liability"] = df.apply(
        _expected_liability_row, axis=1, n_lifelines=n_lifelines
    )

    # Aggregate
    return df.groupby(season_col).agg(
        worst_case_bets=("liability", "count"),
        worst_case_turnover=("turnover_total", "sum"),
        worst_case_payout=("liability", "sum"),
        expected_payout=("expected_liability", "sum"),
    )


if __name__ == "__main__":

    leg_df = pd.read_parquet("data/sgm_leg_data_results.parquet")

    leg_df[["DI", "TK", "MK", "GL"]] = leg_df[["DI", "TK", "MK", "GL"]].fillna(0)

    # See how many things (disposals, tackles, goals) a leg missed by
    leg_df["is_disposal_leg"] = leg_df["stat"].eq("disposals")

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
    disp_only = disp_only.sort_values(["bet_id", "miss_by"], ascending=[True, False])

    # this makes an ordered list of how many disposals each leg lost by, grouped at the bet level
    # It's a ragged list of
    disp_all_list = (
        disp_only.groupby("bet_id")["miss_by"]
        .apply(list)
        .rename("disp_miss_by_including_wins")
    )

    # Make our overall bets table by combining the legs
    bets = leg_df.groupby(["bet_id"]).agg(
        season=("year", "first"),
        event_date=("date", "first"),
        turnover_bonus_bet=("turnover_bonus_bet", "sum"),
        turnover_cash=("turnover_cash", "sum"),
        turnover_total=("turnover_total", "sum"),
        bet_fixed_odds=("bet_fixed_odds", "first"),
        payout=("payout", "sum"),
        net_win=("net_win", "sum"),
        n_legs=("sgm_leg_count", "max"),
        n_disp_legs=("is_disposal_leg", "sum"),
        n_winning_legs=("leg_won", "sum"),
        disp_leg_missed_by_1=("disposals_missed_by_1", "sum"),
        disp_leg_missed_by_2=("disposals_missed_by_2", "sum"),
        disp_leg_missed_by_3=("disposals_missed_by_3", "sum"),
        disp_leg_missed_by_4=("disposals_missed_by_4", "sum"),
        disp_leg_missed_by_5=("disposals_missed_by_5", "sum"),
    )
    bets["bet_won"] = bets["payout"] < 0
    bets["lost_by_one_leg"] = bets["n_legs"] - bets["n_winning_legs"] == 1
    bets["liability"] = bets["turnover_total"] * bets["bet_fixed_odds"].astype(float)
    bets["n_losing_legs"] = bets["n_legs"] - bets["n_winning_legs"]

    # Merge onto the list of how many misses each disposal leg had
    bets = bets.merge(disp_all_list, how="left", left_index=True, right_index=True)

    vals = {
        "n_legs": [4, 6, 8, 6, 8, 10],
        "n_lifelines": [1, 2, 3],
        "lifeline_size": [1, 2, 3, 4, 5],
    }
    combinations = pd.MultiIndex.from_product(
        vals.values(), names=vals.keys()
    ).to_frame(index=False)

    results = []
    for index, row in combinations.iterrows():
        tmp = lifeline_summary(
            bets,
            n_legs=row.n_legs,
            n_lifelines=row.n_lifelines,
            disposal_missed_by_column=f"disp_leg_missed_by_{row.lifeline_size}",
        )
        tmp.insert(0, "n_legs", row.n_legs)
        tmp.insert(1, "n_lifelines", row.n_lifelines)
        tmp.insert(2, "lifeline_extra_disposals", row.lifeline_size)
        results.append(tmp)
    final = pd.concat(results, axis=0)
    final.loc[2025].to_csv("2025_lifeline_data.csv")

    # combinations = pd.DataFrame(
    #     data=dict(
    #         n_legs=[4, 4, 6, 6, 8, 8, 6, 6, 8, 8, 10, 10],
    #         n_lifelines=[1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
    #         disposal_missed_by_column=[
    #             "disp_leg_missed_by_one",
    #             "disp_leg_missed_by_two",
    #             "disp_leg_missed_by_five",
    #         ],
    #         *6,
    #         lifeline_size=[1, 2] * 6,
    #     )
    # )
