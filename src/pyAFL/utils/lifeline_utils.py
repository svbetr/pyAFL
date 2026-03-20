import pandas as pd
import numpy as np
import math
import bisect
from typing import List

MAX_BUCKET = 10  # Largest number of disposals we could save


def nPk(n, k):
    return 0 if k > n or k < 0 else math.prod(range(n, n - k, -1))


def would_win_with_lifelines(
    disp_missed_by_list: List[int],
    lifelines: List[int],
    unsaveable_losses: int = 0,  # e.g. losing non-disposal legs
) -> bool:
    """
    disp_missed_by_list: positive integers for losing *disposal* legs, ideally sorted desc (e.g. [5,3,1]).
    lifelines: each +N can save one leg with miss <= N (e.g. [2,3,5]).
    unsaveable_losses: count of other losing legs that lifelines cannot save (default 0).

    Returns True if all losses can be covered, else False.
    """
    # Keep only positive misses and sort largest->smallest (greedy optimal)
    if np.isfinite(disp_missed_by_list).sum() == 0:
        return False
    misses = sorted((m for m in disp_missed_by_list if m > 0), reverse=True)
    lifes = sorted(lifelines, reverse=True)

    # Assign each lifeline to the largest miss it can cover
    for L in lifes:
        # find first miss <= L (since misses are desc, scan from left)
        for i, m in enumerate(misses):
            if m <= L:
                del misses[i]  # saved this leg
                break  # move to next lifeline

    # any remaining misses + unsaveable losses means the bet still loses
    return (len(misses) + int(unsaveable_losses)) == 0


def _expected_liability_row(row, lifelines: List[int]):

    return row["liability"] * success_probability(
        row.wildcard_miss_by_including_wins, lifelines
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
    """
    misses: list like [5,3,2,0,0] (0 = winning leg)
    lifelines: list like [5,4,3,2,1] (identities distinct; order matters)
    """
    misses = np.atleast_1d(misses)

    if not np.all(np.isfinite(misses)):
        return 0.0

    n = len(misses)
    k = len(lifelines)
    if k == 0:
        return 1.0 if all(m <= 0 for m in misses) else 0.0

    # losing legs
    t = sorted([m for m in misses if m > 0])  # ascending
    L = len(t)
    if L > k:  # not enough lifelines
        return 0.0

    v = sorted(lifelines)  # ascending

    # count injective assignments from lifelines -> losing legs that meet thresholds
    ways_cover = 1
    used = 0
    for miss in t:  # smallest miss first
        # lifelines with value >= miss
        ge = len(v) - bisect.bisect_left(v, miss)
        choices = ge - used
        if choices <= 0:
            return 0.0
        ways_cover *= choices
        used += 1

    # order remaining (k-L) lifelines onto any (k-L) distinct non-losing legs
    ways_rest = nPk(n - L, k - L)

    successes = ways_cover * ways_rest
    total = nPk(n, k)
    return successes / total if total else 0.0


def lifeline_summary(
    bets,
    n_legs: int,
    lifelines: List[int],
    season_col: str = "season",
):
    """
    Filters bets for an L-leg, K-lifeline product where the leg missed by
    1..K legs, computes expected liability, and aggregates by season.

    Note you can change whether you want the lifeline to be +1, +2, etc by changing the 'disposal missed by' column name.
    """
    # Base mask: correct product and the bet lost
    base = bets.loc[
        (bets.n_legs.eq(n_legs)) & (~bets.bet_won) & (bets.all_losing_are_wildcard)
    ].copy()

    # Accept any "missed by m" with m in 1..K (and optionally enforce wins == L - m)
    # This loops over the cases where bets miss by 1 for 1 lifeline, 1 or 2 for 2 lifelines, etc
    mask = base["wildcard_miss_by_including_wins"].apply(
        would_win_with_lifelines, lifelines=lifelines
    )

    base["would_win_with_lifelines"] = mask
    df = base.loc[mask].copy()

    bet_save_fraction = base.groupby("season").agg(
        fraction_saved=("would_win_with_lifelines", "mean"),
        N_bets_total=("event_country", "count"),
    )

    if len(df) > 0:

        # Expected liability per row (pass the extra arg via apply kwargs)
        df["expected_liability"] = df.apply(
            _expected_liability_row, axis=1, lifelines=lifelines
        )

        # Aggregate
        agg = (
            df.groupby(season_col)
            .agg(
                worst_case_bets=("liability", "count"),
                worst_case_turnover=("turnover_total", "sum"),
                worst_case_payout=("liability", "sum"),
                expected_payout=("expected_liability", "sum"),
                rpf_fees=("rpf_fee", "sum"),
                poc_fees=("poc_fee", "sum"),
                gross_win_of_original_bets=("gross_win", "sum"),
            )
            .reindex(base[season_col].unique())
            .fillna(0.0)
        )

        return agg.join(bet_save_fraction)
    else:
        index = base[season_col].unique()
        zeros = [0] * len(index)
        return pd.DataFrame(
            data=dict(
                worst_case_bets=zeros,
                worst_case_turnover=zeros,
                worst_case_payout=zeros,
                expected_payout=zeros,
                rpf_fees=zeros,
                poc_fees=zeros,
                gross_win_of_original_bets=zeros,
                fraction_saved=zeros,
            ),
            index=index,
        )
