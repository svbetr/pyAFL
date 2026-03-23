import pandas as pd
from math import comb


def would_win_with_n_activations(
    n_saveable_losing: int,
    n_unsaveable_losing: int,
    n_activations: int,
) -> bool:
    """
    Returns True if a bet would be saved by activating n_activations lifelines.

    n_saveable_losing:   losing wildcard legs whose n_missed_by <= their lifeline_value
    n_unsaveable_losing: losing wildcard legs whose n_missed_by > their lifeline_value
    n_activations:       how many lifelines the customer can turn on
    """
    return n_unsaveable_losing == 0 and n_saveable_losing <= n_activations


def activation_success_probability(
    n_wildcard_legs: int,
    n_saveable_losing: int,
    n_unsaveable_losing: int,
    n_activations: int,
) -> float:
    """
    Probability that randomly activating n_activations lifelines (chosen uniformly
    from all n_wildcard_legs eligible legs) covers every saveable losing leg.

    Returns 0.0 if the bet cannot be saved regardless of placement.
    """
    if n_unsaveable_losing > 0 or n_saveable_losing > n_activations:
        return 0.0
    if n_wildcard_legs < n_activations or n_wildcard_legs == 0:
        return 0.0
    # P(all n_saveable_losing legs fall within the n_activations chosen at random)
    return (
        comb(n_wildcard_legs - n_saveable_losing, n_activations - n_saveable_losing)
        / comb(n_wildcard_legs, n_activations)
    )


def lifeline_summary(
    bets: pd.DataFrame,
    n_legs: int,
    n_activations: int,
    season_col: str = "season",
) -> pd.DataFrame:
    """
    For bets with n_legs legs, computes the expected cost of offering n_activations
    lifelines, where each leg already has a per-leg lifeline_value and has been
    pre-classified as saveable/unsaveable.

    Required bets columns:
        n_legs, bet_won, all_losing_are_wildcard,
        n_wildcard_legs, n_saveable_losing_legs, n_unsaveable_losing_legs,
        liability, turnover_total, rpf_fee, poc_fee, gross_win, season
    """
    base = bets.loc[
        bets.n_legs.eq(n_legs) & ~bets.bet_won & bets.all_losing_are_wildcard
    ].copy()

    base["would_win"] = base.apply(
        lambda r: would_win_with_n_activations(
            r.n_saveable_losing_legs, r.n_unsaveable_losing_legs, n_activations
        ),
        axis=1,
    )

    bet_save_fraction = base.groupby(season_col).agg(
        fraction_saved=("would_win", "mean"),
        N_bets_total=("event_country", "count"),
    )

    df = base.loc[base["would_win"]].copy()

    if len(df) > 0:
        df["expected_liability"] = df.apply(
            lambda r: r.liability * activation_success_probability(
                r.n_wildcard_legs, r.n_saveable_losing_legs,
                r.n_unsaveable_losing_legs, n_activations,
            ),
            axis=1,
        )

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
                N_bets_total=zeros,
            ),
            index=index,
        )
