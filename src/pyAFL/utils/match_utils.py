import pandas as pd
import re


def consistent_name(s: str) -> str:

    s = s.lower().strip()
    s = re.sub(r"[^\w]+", "-", s)

    return re.sub(r"-+", "-", s).strip("-")


def flatten_qcols(df):
    """Flatten a (side_kind, Q#) MultiIndex to 'q#_side_kind' lowercase."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [f"{q.lower()}_{side.lower()}" for side, q in df.columns]
    else:
        # already flat: just normalise to lowercase
        df = df.rename(columns={c: c.lower() for c in df.columns})
    return df


def scoring_points_per_quarter(
    df: pd.DataFrame,
    for_col: str = "Scoring_For",
    against_col: str = "Scoring_Against",
    n_quarters: int = 4,
    return_increments: bool = True,
) -> pd.DataFrame:
    """
    Parse 'G.B G.B G.B G.B' scoring strings into per-quarter points.
    Returns a DataFrame ready to .join() back to df, with columns like:
      ('For','Q1'), ('For','Q2'), ..., ('Against','Q1'), ...
    If return_increments=True, also returns ('For_inc','Qk') / ('Against_inc','Qk')
    for per-quarter points (Q1_inc = Q1; Qk_inc = Qk - Q{k-1}).
    """

    pat = r"(\d+)\.(\d+)"  # goals.behinds

    def side_points(s: pd.Series) -> pd.DataFrame:
        # Extract all goals/behinds pairs; index=(row, match#)
        ex = (
            s.fillna("").str.extractall(pat).astype("Int64")
        )  # two cols: goals, behinds
        if ex.empty:
            # create empty frame with the right index
            out = pd.DataFrame(
                index=s.index,
                columns=[f"Q{i}" for i in range(1, n_quarters + 1)],
                dtype="Int64",
            )
            return out

        ex.columns = ["goals", "behinds"]
        ex["points"] = ex["goals"] * 6 + ex["behinds"]

        # Unstack match# → columns 0..k-1 (cumulative by quarter)
        pts = ex["points"].unstack(level=1)

        # Ensure exactly n_quarters columns, pad with NA if fewer
        # (columns are integers 0..k-1 after unstack)
        cols_wanted = list(range(n_quarters))
        pts = pts.reindex(columns=cols_wanted)

        # Rename to Q1..Qn
        pts.columns = [f"Q{i}" for i in range(1, n_quarters + 1)]

        # Keep nullable integer dtype
        return pts.astype("Int64")

    # Compute cumulative points per quarter
    for_pts = side_points(df[for_col])
    ag_pts = side_points(df[against_col])

    # Build MultiIndex columns
    for_pts.columns = pd.MultiIndex.from_product([["For_cumulative"], for_pts.columns])
    ag_pts.columns = pd.MultiIndex.from_product(
        [["Against_cumulative"], ag_pts.columns]
    )

    out = pd.concat([for_pts, ag_pts], axis=1)

    if return_increments:
        # Per-quarter increments: Q1_inc = Q1; Qk_inc = Qk - Q{k-1}
        def increments(pts: pd.DataFrame) -> pd.DataFrame:
            inc = pts.diff(axis=1)
            first_col = pts.iloc[:, 0]
            inc.iloc[:, 0] = first_col
            return inc.astype("Int64")

        for_inc = increments(for_pts.droplevel(0, axis=1))
        ag_inc = increments(ag_pts.droplevel(0, axis=1))

        for_inc.columns = pd.MultiIndex.from_product([["For_inc"], for_inc.columns])
        ag_inc.columns = pd.MultiIndex.from_product([["Against_inc"], ag_inc.columns])

        out = pd.concat([out, for_inc, ag_inc], axis=1)

    return out


def for_against_to_t1_t2(qdf: pd.DataFrame, is_team_as_t1: pd.Series) -> pd.DataFrame:
    """
    Given flat columns like 'q1_for_cumulative', 'q1_against_cumulative', …
    return a DataFrame with columns 'q1_t1_cumulative', 'q1_t2_cumulative', …
    mapping For→t1 and Against→t2 if the row's team is team_1, else swapped.
    """
    qdf = qdf.copy()

    # Ensure lowercase names for regex pairing
    qdf.columns = [c.lower() for c in qdf.columns]

    # Find all quarter+kind combos (e.g., 'q1_cumulative', 'q2_inc', …)
    pat = re.compile(r"^(q\d+)_(for|against)_(\w+)$")
    triples = [pat.match(c).groups() for c in qdf.columns if pat.match(c)]
    uniq_qk = sorted({(q, kind) for q, side, kind in triples})

    out_cols = {}
    for q, kind in uniq_qk:
        cf = f"{q}_for_{kind}"
        ca = f"{q}_against_{kind}"
        if cf not in qdf.columns or ca not in qdf.columns:
            continue  # skip incomplete pairs

        # team_1 values come from For if the current row's team == team_1, else from Against
        t1 = qdf[cf].where(is_team_as_t1, qdf[ca])
        t2 = qdf[ca].where(is_team_as_t1, qdf[cf])

        out_cols[f"{q}_t1_{kind}"] = t1
        out_cols[f"{q}_t2_{kind}"] = t2

    return pd.DataFrame(out_cols, index=qdf.index)
