import re
import numpy as np
import pandas as pd
import operator
from pyAFL.utils.match_utils import consistent_name


def evaluate_sgm_legs(
    legs: pd.DataFrame,
    team_match_stats: pd.DataFrame,
    player_match_stats: pd.DataFrame,
    keep_markets=None,
    leg_id_col: str = "bet_leg_id",
):
    """
    Evaluate SGM legs against team/player facts.

    Required columns in `legs`:
      - matchID, event, outcome
      - homeTeam, awayTeam, homeTeamID, awayTeamID
      - (optional) playerID for player markets
      - a unique id column per leg (default 'bet_leg_id')

    team_match_stats: ['matchID','teamID','finalScore','halftimeScore', ...]
    player_match_stats: ['matchID','playerId','tries', ...]

    Returns: DataFrame = legs + canonical columns + 'actual' + 'won'
    """
    if keep_markets is not None:
        legs = legs[legs["event"].isin(keep_markets)].copy()
    else:
        legs = legs.copy()

    # Canonicalise all legs
    print("Reformatting all legs consistently...")
    canon = legs.apply(normalize_leg, axis=1, result_type="expand")
    out = pd.concat([legs, canon], axis=1).copy()
    out = out.reset_index(drop=True)
    print("Done! Total legs to evaluate:", len(out))

    # ---------------- evaluations ----------------
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        "=": operator.eq,
        ">=": operator.ge,
        ">": operator.gt,
    }

    # Player goals
    part = out.query("subject_type=='player' and stat=='tries'").copy()
    print(f"Evaluating {len(part)} player tryscorer legs...")
    if not part.empty:
        pstats = player_match_stats.drop_duplicates(
            subset=["gameID", "playerId"]
        ).set_index(["gameID", "playerId"])[["DI"]]
        part = part.join(pstats, on=["gameID", "subject_id"])

        part["actual"] = part["tries"].fillna(0)
        part["won"] = [
            ops[c](a, t) if pd.notna(a) and c in ops else np.nan
            for c, a, t in zip(part["comparator"], part["actual"], part["threshold"])
        ]

        out.loc[part.index, ["actual", "won"]] = part[["actual", "won"]]

    # Totals (FT + HT)
    legs_tot = out.query("stat=='total_points'").copy()
    print(f"Evaluating {len(legs_tot)} totals legs...")
    if not legs_tot.empty:
        totals = team_match_stats.groupby("matchID")[
            ["totalPoints", "halfTimeTotalPoints"]
        ].sum()

        # join preserves legs_tot's original index/order
        legs_tot = legs_tot.join(totals, on="matchID")
        legs_tot["actual"] = np.where(
            legs_tot["scope"].eq("1H"),
            legs_tot["halfTimeTotalPoints"],
            legs_tot["totalPoints"],
        )
        legs_tot["won"] = [
            ops[c](a, t) if pd.notna(a) and c in ops else np.nan
            for c, a, t in zip(
                legs_tot["comparator"], legs_tot["actual"], legs_tot["threshold"]
            )
        ]
        out.loc[legs_tot.index, ["actual", "won"]] = legs_tot[
            ["actual", "won"]
        ].to_numpy()

    # Match result → margin > 0 for chosen team
    legs_res = out.query(
        "subject_type=='team' and stat in ('margin', 'handicap_pyol')"
    ).copy()
    print(f"Evaluating {len(legs_res)} match result, PYOL and margin legs...")
    if not legs_res.empty:

        m = team_match_stats[
            ["matchID", "homeId", "awayId", "homeScore", "awayScore"]
        ].drop_duplicates(subset=["matchID"])
        # Compute home margin once
        m["home_margin"] = m["homeScore"] - m["awayScore"]

        # Expand to team-level margins: one row for home, one for away (negated)
        m_home = m.rename(columns={"homeId": "team_id"})[
            ["matchID", "team_id", "home_margin"]
        ]
        m_home = m_home.rename(columns={"home_margin": "margin"})

        m_away = m.rename(columns={"awayId": "team_id"})[
            ["matchID", "team_id", "home_margin"]
        ]
        m_away = m_away.assign(margin=-m_away["home_margin"])[
            ["matchID", "team_id", "margin"]
        ]

        margins = pd.concat([m_home, m_away], ignore_index=True)

        legs_res = legs_res.join(
            margins.set_index(["matchID", "team_id"])["margin"],
            on=["matchID", "subject_id"],
        )

        legs_res["actual"] = legs_res["margin"]

        # Check using our comparisons if the leg won
        legs_res["won"] = [
            (
                (ops[c](a, t) and (np.isnan(tu) or a <= tu))
                if pd.notna(a) and c in ops
                else np.nan
            )
            for c, a, t, tu in zip(
                legs_res["comparator"],
                legs_res["actual"],
                legs_res["threshold"],
                legs_res["threshold_upper"],
            )
        ]

        out.loc[legs_res.index, ["actual", "won"]] = legs_res[
            ["actual", "won"]
        ].to_numpy()

    # Half-Time / Full-Time
    legs_hf = out.query("stat=='ht_ft'").copy()
    print(f"Evaluating {len(legs_hf)} HT/FT legs...")
    if not legs_hf.empty:

        m = team_match_stats[
            [
                "matchID",
                "homeId",
                "awayId",
                "homeScore",
                "awayScore",
                "homeScoreHalfTime",
                "awayScoreHalfTime",
            ]
        ].drop_duplicates(subset=["matchID"])

        # Score diffs
        ht_diff = m["homeScoreHalfTime"] - m["awayScoreHalfTime"]
        ft_diff = m["homeScore"] - m["awayScore"]

        # Vectorized winners: home if diff>0, away if diff<0, 0 if draw
        winners = pd.DataFrame(
            {
                "matchID": m["matchID"],
                "ht_winner": np.where(
                    ht_diff > 0, m["homeId"], np.where(ht_diff < 0, m["awayId"], 0)
                ),
                "ft_winner": np.where(
                    ft_diff > 0, m["homeId"], np.where(ft_diff < 0, m["awayId"], 0)
                ),
            }
        )

        # Join preserves legs_hf’s original index and only adds the two columns
        legs_hf = legs_hf.join(winners.set_index("matchID"), on="matchID")
        legs_hf["actual_tuple"] = list(zip(legs_hf["ht_winner"], legs_hf["ft_winner"]))
        legs_hf["expected_tuple"] = list(
            zip(legs_hf["expected_ht"], legs_hf["expected_ft"])
        )
        legs_hf["won"] = legs_hf["actual_tuple"] == legs_hf["expected_tuple"]
        out.loc[legs_hf.index, "actual"] = legs_hf["actual_tuple"]
        out.loc[legs_hf.index, "won"] = legs_hf["won"]

    # Tidy up types (nullable ints are handy)
    for col in ["subject_id", "expected_ht", "expected_ft"]:
        if col in out:
            out[col] = out[col].astype("Int64")

    print(f"Evaluated {len(out)} legs.")
    return out


# ---------------- helpers ----------------
def parse_over_under_strings(s: str):
    m = re.search(r"^\s*(Over|Under)\s+([0-9]+(?:\.[0-9]+)?)", str(s), re.I)
    if not m:
        return None, np.nan
    side = m.group(1).lower()
    line = float(m.group(2))
    return side, line


def map_team(term, row):
    t = consistent_name(term)
    home_name, away_name = consistent_name(row["team_1"]), consistent_name(
        row["team_2"]
    )
    if t in {"home", "h", home_name}:
        return int(row["team1_ID"])
    if t in {"away", "a", away_name}:
        return int(row["team2_ID"])
    if t in {"draw", "tie"}:
        return 0  # sentinel for draw
    # try exact name match as fallback
    if t == home_name:
        return int(row["team1_ID"])
    if t == away_name:
        return int(row["team2_ID"])
    return np.nan


def normalize_leg(row):
    market = str(row["event_class"]).strip()
    out = str(row["outcome"]).strip()
    base = dict(
        subject_type=None,
        subject_id=np.nan,
        stat=None,
        comparator=None,
        threshold=np.nan,
        threshold_upper=np.nan,
        scope="match",
        expected_ht=np.nan,
        expected_ft=np.nan,
    )

    # Normalize all our legs into the base format above
    subject_type_dict = {
        "Player Totals - Disposals": "player",
        "Player Totals - Goals": "player",
        "Anytime Goalscorer": "player",
        "Draw No Bet": "team",
        "Total Match Points": "points",
        "Player marks": "player",
        "Player tackles": "player",
        "Half-time/Full-time": "team",
        "1st Quarter Result (3-way)": "team",
        "Half-time Result": "team",
        "Winning Margin": "margin",
    }

    sub_type = subject_type_dict.get(market, None)

    if sub_type is None:
        return base

    # Player stats
    if sub_type == "player":

        stat_dict = {
            "Player Totals - Disposals": "disposals",
            "Player Totals - Goals": "goals",
            "Anytime Goalscorer": "goals",
            "Player marks": "marks",
            "Player tackles": "tackles",
        }

        subj = row.get("playerID")
        # Turn out outcome row into a float if it matches "[number]+".
        # If it doesn't (i.e. it's a name) then return 1.0 instead
        th = (
            float(re.sub(r"\+$", "", row.outcome))
            if re.fullmatch(r"\s*\d+\+?\s*", row.outcome)
            else 1.0
        )
        # some sources store playerID as float; cast safely
        subj = int(subj) if pd.notna(subj) else np.nan
        return {
            **base,
            "subject_type": "player",
            "subject_id": subj,
            "stat": stat_dict[market],
            "comparator": ">=",
            "threshold": float(th),
        }

    # Match Result (team to win) → margin > 0
    if market == "Draw No Bet":

        # Get the team ID
        tid = map_team(out, row)

        # Set up the Handicap values
        if row.bet_detail_type_code == "HC":

            threshold = row.pyol_points
            return {
                **base,
                "subject_type": "team",
                "subject_id": tid,
                "stat": "handicap_pyol",
                "comparator": ">",
                "threshold": threshold,
            }
        elif row.bet_detail_type_code == "M1-39":

            # Need something here to say that margin must be between 1 and 39

            return {
                **base,
                "subject_type": "team",
                "subject_id": tid,
                "stat": "margin",
                "comparator": ">",
                "threshold": 0.0,
                "threshold_upper": 39,
            }
        elif row.bet_detail_type_code == "M40+":

            # Need something here to say that margin must be between 1 and 39

            return {
                **base,
                "subject_type": "team",
                "subject_id": tid,
                "stat": "margin",
                "comparator": ">=",
                "threshold": 40.0,
            }

        elif row.bet_detail_type_code == "WIN":
            return {
                **base,
                "subject_type": "team",
                "subject_id": tid,
                "stat": "margin",
                "comparator": ">",
                "threshold": 0.0,
            }

    # Totals (full-time)
    if market == "Total Match Points":
        side, line = parse_over_under_strings(out)
        comp = ">" if side == "over" else "<" if side == "under" else None
        return {
            **base,
            "subject_type": "match",
            "stat": "total_points",
            "comparator": comp,
            "threshold": line,
            "scope": "match",
        }

    # Half-Time/Full-Time
    if market == "Half-time/Full-time":
        p = [x.strip() for x in out.split("/")[:2]]
        ht = map_team(p[0], row) if len(p) > 0 else np.nan
        ft = map_team(p[1], row) if len(p) > 1 else np.nan
        return {
            **base,
            "subject_type": "match",
            "stat": "ht_ft",
            "expected_ht": ht,
            "expected_ft": ft,
        }

    return base
