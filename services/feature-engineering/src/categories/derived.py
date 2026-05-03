"""Derived features — ~95 features computed from other categories' outputs.

Includes Elo ratings, Poisson predictions, momentum composites, interaction
terms, and cross-category correlations. Runs in phase 2 after all base
categories have been computed.
"""

import logging
import math
from typing import Dict, Optional

import numpy as np

from ..core.registry import FeatureMetadata
from ..utils.math_helpers import (
    elo_expected,
    poisson_match_probabilities,
    safe_divide,
)
from .base import BaseFeatureComputer, MatchContext

logger = logging.getLogger(__name__)


class DerivedFeatureComputer(BaseFeatureComputer):
    """Compute derived/composite features from prior feature outputs."""

    category = "derived"

    def _register_features(self) -> None:
        defs = []

        # Elo ratings
        for side in ["home", "away"]:
            defs.append(FeatureMetadata(
                name=f"derived__elo__{side}_rating",
                category=self.category,
                description=f"{side} team Elo rating",
            ))
        defs.extend([
            FeatureMetadata(
                name="derived__elo__diff",
                category=self.category,
                description="Elo rating difference (home - away)",
            ),
            FeatureMetadata(
                name="derived__elo__home_expected",
                category=self.category,
                description="Elo-based expected score for home team",
                min_value=0, max_value=1,
            ),
        ])

        # Poisson model
        defs.extend([
            FeatureMetadata(
                name="derived__poisson__home_lambda",
                category=self.category,
                description="Poisson expected goals for home team",
                min_value=0,
            ),
            FeatureMetadata(
                name="derived__poisson__away_lambda",
                category=self.category,
                description="Poisson expected goals for away team",
                min_value=0,
            ),
            FeatureMetadata(
                name="derived__poisson__home_win",
                category=self.category,
                description="Poisson home win probability",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__poisson__draw",
                category=self.category,
                description="Poisson draw probability",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__poisson__away_win",
                category=self.category,
                description="Poisson away win probability",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__poisson__over25",
                category=self.category,
                description="Poisson probability of over 2.5 goals",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__poisson__btts",
                category=self.category,
                description="Poisson probability of BTTS",
                min_value=0, max_value=1,
            ),
        ])

        # Momentum composites
        for side in ["home", "away"]:
            defs.extend([
                FeatureMetadata(
                    name=f"derived__momentum__{side}__attack",
                    category=self.category,
                    description=f"{side} attack momentum (goals + xG trend)",
                ),
                FeatureMetadata(
                    name=f"derived__momentum__{side}__defense",
                    category=self.category,
                    description=f"{side} defensive momentum",
                ),
                FeatureMetadata(
                    name=f"derived__momentum__{side}__overall",
                    category=self.category,
                    description=f"{side} overall momentum composite",
                ),
                FeatureMetadata(
                    name=f"derived__momentum__{side}__form_index",
                    category=self.category,
                    description=f"{side} weighted form index (recent > older)",
                ),
            ])

        # Momentum comparison
        defs.extend([
            FeatureMetadata(
                name="derived__momentum__attack_diff",
                category=self.category,
                description="Attack momentum difference (home - away)",
            ),
            FeatureMetadata(
                name="derived__momentum__defense_diff",
                category=self.category,
                description="Defense momentum difference (home - away)",
            ),
            FeatureMetadata(
                name="derived__momentum__overall_diff",
                category=self.category,
                description="Overall momentum difference (home - away)",
            ),
        ])

        # Interaction terms (cross-category)
        defs.extend([
            FeatureMetadata(
                name="derived__interact__home_form_x_h2h",
                category=self.category,
                description="Home form PPG * H2H home win rate",
            ),
            FeatureMetadata(
                name="derived__interact__away_form_x_h2h",
                category=self.category,
                description="Away form PPG * H2H away win rate",
            ),
            FeatureMetadata(
                name="derived__interact__home_attack_x_away_defense",
                category=self.category,
                description="Home xG * Away goals conceded",
            ),
            FeatureMetadata(
                name="derived__interact__away_attack_x_home_defense",
                category=self.category,
                description="Away xG * Home goals conceded",
            ),
            FeatureMetadata(
                name="derived__interact__form_x_rest_home",
                category=self.category,
                description="Home form * rest advantage",
            ),
            FeatureMetadata(
                name="derived__interact__form_x_rest_away",
                category=self.category,
                description="Away form * rest advantage",
            ),
            FeatureMetadata(
                name="derived__interact__odds_x_form_home",
                category=self.category,
                description="Home implied prob * form PPG",
            ),
            FeatureMetadata(
                name="derived__interact__odds_x_form_away",
                category=self.category,
                description="Away implied prob * form PPG",
            ),
            FeatureMetadata(
                name="derived__interact__position_x_form",
                category=self.category,
                description="Position diff * form diff",
            ),
            FeatureMetadata(
                name="derived__interact__injuries_x_form_home",
                category=self.category,
                description="Home unavailability * inverse form",
            ),
            FeatureMetadata(
                name="derived__interact__injuries_x_form_away",
                category=self.category,
                description="Away unavailability * inverse form",
            ),
        ])

        # Strength differentials
        for metric in [
            "xg", "shots", "possession", "ppg", "goals_scored",
            "goals_conceded", "clean_sheets",
        ]:
            defs.append(FeatureMetadata(
                name=f"derived__diff__{metric}",
                category=self.category,
                description=f"Differential in {metric} (home - away, last 5)",
            ))

        # Relative to league average
        for side in ["home", "away"]:
            for metric in ["goals_scored", "goals_conceded", "xg", "xg_against"]:
                defs.append(FeatureMetadata(
                    name=f"derived__relative__{side}__{metric}",
                    category=self.category,
                    description=f"{side} {metric} relative to league avg",
                ))

        # Value-based features
        defs.extend([
            FeatureMetadata(
                name="derived__value__squad_ratio",
                category=self.category,
                description="Home/Away squad value ratio",
                min_value=0,
            ),
            FeatureMetadata(
                name="derived__value__squad_diff_log",
                category=self.category,
                description="Log of squad value difference",
            ),
        ])

        # Model agreement
        defs.extend([
            FeatureMetadata(
                name="derived__agreement__elo_vs_odds",
                category=self.category,
                description="Agreement between Elo and odds (1 - abs diff)",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__agreement__poisson_vs_odds",
                category=self.category,
                description="Agreement between Poisson and odds",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__agreement__form_vs_odds",
                category=self.category,
                description="Agreement between form and odds",
                min_value=0, max_value=1,
            ),
        ])

        # Composite scores
        defs.extend([
            FeatureMetadata(
                name="derived__composite__home_strength",
                category=self.category,
                description="Composite home team strength score",
            ),
            FeatureMetadata(
                name="derived__composite__away_strength",
                category=self.category,
                description="Composite away team strength score",
            ),
            FeatureMetadata(
                name="derived__composite__match_quality",
                category=self.category,
                description="Expected match quality/entertainment index",
                min_value=0,
            ),
            FeatureMetadata(
                name="derived__composite__upset_potential",
                category=self.category,
                description="Likelihood of an upset",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__composite__goals_expectation",
                category=self.category,
                description="Composite expected total goals",
                min_value=0,
            ),
        ])

        # Confidence features
        defs.extend([
            FeatureMetadata(
                name="derived__confidence__data_completeness",
                category=self.category,
                description="Fraction of non-null input features",
                min_value=0, max_value=1,
            ),
            FeatureMetadata(
                name="derived__confidence__model_consensus",
                category=self.category,
                description="How much Elo, Poisson, odds agree",
                min_value=0, max_value=1,
            ),
        ])

        self._feature_defs = defs

    def compute(
        self, ctx: MatchContext, prior_features: Optional[Dict[str, Optional[float]]] = None, **kwargs
    ) -> Dict[str, Optional[float]]:
        features = self._empty_features()
        pf = prior_features or {}

        self._compute_elo(features, pf)
        self._compute_poisson(features, pf, ctx)
        self._compute_momentum(features, pf)
        self._compute_interactions(features, pf)
        self._compute_differentials(features, pf)
        self._compute_relative_to_league(features, pf)
        self._compute_value_features(features, pf)
        self._compute_agreement(features, pf)
        self._compute_composites(features, pf)
        self._compute_confidence(features, pf)

        return features

    # ---------- Elo ----------

    def _compute_elo(self, features, pf):
        # Approximate Elo from form PPG (simplified without full history)
        home_ppg = pf.get("team__home__form__points_per_game__last10")
        away_ppg = pf.get("team__away__form__points_per_game__last10")

        if home_ppg is not None:
            home_elo = 1500 + (home_ppg - 1.5) * 200
            features["derived__elo__home_rating"] = home_elo
        if away_ppg is not None:
            away_elo = 1500 + (away_ppg - 1.5) * 200
            features["derived__elo__away_rating"] = away_elo

        home_elo = features.get("derived__elo__home_rating")
        away_elo = features.get("derived__elo__away_rating")
        if home_elo is not None and away_elo is not None:
            features["derived__elo__diff"] = home_elo - away_elo
            features["derived__elo__home_expected"] = elo_expected(
                home_elo, away_elo
            )

    # ---------- Poisson ----------

    def _compute_poisson(self, features, pf, ctx):
        # Home lambda = home attack * away defense weakness
        home_xg = pf.get("team__home__form__avg_xg__last5")
        away_xg = pf.get("team__away__form__avg_xg__last5")
        home_xga = pf.get("team__home__form__avg_xg_against__last5")
        away_xga = pf.get("team__away__form__avg_xg_against__last5")
        league_avg = pf.get("ctx__league__avg_goals")

        if all(v is not None for v in [home_xg, away_xga, league_avg]) and league_avg > 0:
            league_avg_per_side = league_avg / 2.0
            home_attack = safe_divide(home_xg, league_avg_per_side, 1.0)
            away_defense = safe_divide(away_xga, league_avg_per_side, 1.0)
            home_lambda = home_attack * away_defense * league_avg_per_side
            features["derived__poisson__home_lambda"] = home_lambda
        else:
            home_lambda = home_xg if home_xg else None

        if all(v is not None for v in [away_xg, home_xga, league_avg]) and league_avg > 0:
            league_avg_per_side = league_avg / 2.0
            away_attack = safe_divide(away_xg, league_avg_per_side, 1.0)
            home_defense = safe_divide(home_xga, league_avg_per_side, 1.0)
            away_lambda = away_attack * home_defense * league_avg_per_side
            features["derived__poisson__away_lambda"] = away_lambda
        else:
            away_lambda = away_xg if away_xg else None

        if home_lambda is not None and away_lambda is not None:
            h_lam = features.get("derived__poisson__home_lambda", home_lambda)
            a_lam = features.get("derived__poisson__away_lambda", away_lambda)
            p_home, p_draw, p_away = poisson_match_probabilities(h_lam, a_lam)
            features["derived__poisson__home_win"] = p_home
            features["derived__poisson__draw"] = p_draw
            features["derived__poisson__away_win"] = p_away

            # Over 2.5 = 1 - P(0,1,2 total goals)
            h_lam + a_lam
            from ..utils.math_helpers import poisson_probability

            p_under25 = 0.0
            for i in range(3):  # 0, 1, 2 total goals
                for hi in range(i + 1):
                    ai = i - hi
                    p_under25 += poisson_probability(h_lam, hi) * poisson_probability(a_lam, ai)
            features["derived__poisson__over25"] = 1.0 - p_under25

            # BTTS = 1 - P(home=0) - P(away=0) + P(both=0)
            p_home_zero = poisson_probability(h_lam, 0)
            p_away_zero = poisson_probability(a_lam, 0)
            features["derived__poisson__btts"] = (
                1.0 - p_home_zero - p_away_zero + p_home_zero * p_away_zero
            )

    # ---------- Momentum ----------

    def _compute_momentum(self, features, pf):
        for side in ["home", "away"]:
            # Attack momentum: recent goals + xG trend
            gs5 = pf.get(f"team__{side}__form__goals_scored__last5")
            gs3 = pf.get(f"team__{side}__form__goals_scored__last3")
            xg5 = pf.get(f"team__{side}__form__avg_xg__last5")
            pf.get(f"team__{side}__form__avg_xg__last3")

            attack = None
            if gs5 is not None and gs3 is not None and xg5 is not None:
                gs_rate5 = gs5 / 5.0
                gs_rate3 = gs3 / 3.0 if gs3 else 0
                attack = 0.5 * gs_rate3 + 0.3 * gs_rate5 + 0.2 * xg5
                features[f"derived__momentum__{side}__attack"] = attack

            # Defense momentum: fewer goals conceded = better
            gc5 = pf.get(f"team__{side}__form__goals_conceded__last5")
            gc3 = pf.get(f"team__{side}__form__goals_conceded__last3")
            xga5 = pf.get(f"team__{side}__form__avg_xg_against__last5")

            defense = None
            if gc5 is not None and gc3 is not None:
                gc_rate5 = gc5 / 5.0
                gc_rate3 = gc3 / 3.0 if gc3 else 0
                xga = xga5 if xga5 is not None else gc_rate5
                defense = -(0.5 * gc_rate3 + 0.3 * gc_rate5 + 0.2 * xga)
                features[f"derived__momentum__{side}__defense"] = defense

            # Overall
            if attack is not None and defense is not None:
                features[f"derived__momentum__{side}__overall"] = (
                    0.6 * attack + 0.4 * defense
                )

            # Form index (weighted PPG across windows)
            ppg3 = pf.get(f"team__{side}__form__points_per_game__last3")
            ppg5 = pf.get(f"team__{side}__form__points_per_game__last5")
            ppg10 = pf.get(f"team__{side}__form__points_per_game__last10")

            if ppg3 is not None and ppg5 is not None:
                ppg10v = ppg10 if ppg10 is not None else ppg5
                features[f"derived__momentum__{side}__form_index"] = (
                    0.5 * ppg3 + 0.3 * ppg5 + 0.2 * ppg10v
                )

        # Momentum diffs
        for metric in ["attack", "defense", "overall"]:
            hm = features.get(f"derived__momentum__home__{metric}")
            am = features.get(f"derived__momentum__away__{metric}")
            if hm is not None and am is not None:
                features[f"derived__momentum__{metric}_diff"] = hm - am

    # ---------- Interactions ----------

    def _compute_interactions(self, features, pf):
        home_ppg = pf.get("team__home__form__points_per_game__last5")
        away_ppg = pf.get("team__away__form__points_per_game__last5")
        h2h_home_wr = pf.get("h2h__home_win_rate")
        h2h_away_wr = pf.get("h2h__away_win_rate")

        if home_ppg is not None and h2h_home_wr is not None:
            features["derived__interact__home_form_x_h2h"] = home_ppg * h2h_home_wr
        if away_ppg is not None and h2h_away_wr is not None:
            features["derived__interact__away_form_x_h2h"] = away_ppg * h2h_away_wr

        home_xg = pf.get("team__home__form__avg_xg__last5")
        away_gc = pf.get("team__away__form__goals_conceded__last5")
        away_xg = pf.get("team__away__form__avg_xg__last5")
        home_gc = pf.get("team__home__form__goals_conceded__last5")

        if home_xg is not None and away_gc is not None:
            features["derived__interact__home_attack_x_away_defense"] = (
                home_xg * (away_gc / 5.0)
            )
        if away_xg is not None and home_gc is not None:
            features["derived__interact__away_attack_x_home_defense"] = (
                away_xg * (home_gc / 5.0)
            )

        # Form x rest
        home_rest = pf.get("ctx__home__rest_days")
        away_rest = pf.get("ctx__away__rest_days")
        if home_ppg is not None and home_rest is not None:
            rest_factor = min(home_rest / 7.0, 1.5)  # Normalize
            features["derived__interact__form_x_rest_home"] = home_ppg * rest_factor
        if away_ppg is not None and away_rest is not None:
            rest_factor = min(away_rest / 7.0, 1.5)
            features["derived__interact__form_x_rest_away"] = away_ppg * rest_factor

        # Odds x form
        home_prob = pf.get("odds__implied_prob__home")
        away_prob = pf.get("odds__implied_prob__away")
        if home_prob is not None and home_ppg is not None:
            features["derived__interact__odds_x_form_home"] = home_prob * home_ppg
        if away_prob is not None and away_ppg is not None:
            features["derived__interact__odds_x_form_away"] = away_prob * away_ppg

        # Position x form
        pos_diff = pf.get("ctx__position_diff")
        if pos_diff is not None and home_ppg is not None and away_ppg is not None:
            features["derived__interact__position_x_form"] = (
                pos_diff * (home_ppg - away_ppg)
            )

        # Injuries x form
        home_unavail = pf.get("player__home__unavailable_pct")
        away_unavail = pf.get("player__away__unavailable_pct")
        if home_unavail is not None and home_ppg is not None:
            features["derived__interact__injuries_x_form_home"] = (
                home_unavail * (3.0 - home_ppg)
            )
        if away_unavail is not None and away_ppg is not None:
            features["derived__interact__injuries_x_form_away"] = (
                away_unavail * (3.0 - away_ppg)
            )

    # ---------- Differentials ----------

    def _compute_differentials(self, features, pf):
        diff_map = {
            "xg": ("team__home__form__avg_xg__last5", "team__away__form__avg_xg__last5"),
            "shots": ("team__home__form__avg_shots__last5", "team__away__form__avg_shots__last5"),
            "possession": ("team__home__form__avg_possession__last5", "team__away__form__avg_possession__last5"),
            "ppg": ("team__home__form__points_per_game__last5", "team__away__form__points_per_game__last5"),
            "goals_scored": ("team__home__form__goals_scored__last5", "team__away__form__goals_scored__last5"),
            "goals_conceded": ("team__home__form__goals_conceded__last5", "team__away__form__goals_conceded__last5"),
            "clean_sheets": ("team__home__form__clean_sheets__last5", "team__away__form__clean_sheets__last5"),
        }

        for metric, (h_key, a_key) in diff_map.items():
            hv = pf.get(h_key)
            av = pf.get(a_key)
            if hv is not None and av is not None:
                features[f"derived__diff__{metric}"] = hv - av

    # ---------- Relative to League ----------

    def _compute_relative_to_league(self, features, pf):
        league_avg_goals = pf.get("ctx__league__avg_goals")
        if league_avg_goals is None or league_avg_goals == 0:
            return

        half_avg = league_avg_goals / 2.0

        for side in ["home", "away"]:
            gs = pf.get(f"team__{side}__form__goals_scored__last5")
            gc = pf.get(f"team__{side}__form__goals_conceded__last5")
            xg = pf.get(f"team__{side}__form__avg_xg__last5")
            xga = pf.get(f"team__{side}__form__avg_xg_against__last5")

            if gs is not None:
                features[f"derived__relative__{side}__goals_scored"] = (
                    (gs / 5.0) / half_avg
                )
            if gc is not None:
                features[f"derived__relative__{side}__goals_conceded"] = (
                    (gc / 5.0) / half_avg
                )
            if xg is not None:
                features[f"derived__relative__{side}__xg"] = xg / half_avg
            if xga is not None:
                features[f"derived__relative__{side}__xg_against"] = xga / half_avg

    # ---------- Value ----------

    def _compute_value_features(self, features, pf):
        home_val = pf.get("player__home__squad_value_total")
        away_val = pf.get("player__away__squad_value_total")

        if home_val is not None and away_val is not None and away_val > 0:
            features["derived__value__squad_ratio"] = safe_divide(home_val, away_val)
            diff = abs(home_val - away_val)
            features["derived__value__squad_diff_log"] = (
                math.log1p(diff) if diff > 0 else 0.0
            )

    # ---------- Agreement ----------

    def _compute_agreement(self, features, pf):
        odds_home = pf.get("odds__implied_prob__home")

        elo_exp = features.get("derived__elo__home_expected")
        if elo_exp is not None and odds_home is not None:
            features["derived__agreement__elo_vs_odds"] = (
                1.0 - abs(elo_exp - odds_home)
            )

        poisson_home = features.get("derived__poisson__home_win")
        if poisson_home is not None and odds_home is not None:
            features["derived__agreement__poisson_vs_odds"] = (
                1.0 - abs(poisson_home - odds_home)
            )

        # Form vs odds: normalize PPG to probability-like scale
        home_ppg = pf.get("team__home__form__points_per_game__last5")
        away_ppg = pf.get("team__away__form__points_per_game__last5")
        if home_ppg is not None and away_ppg is not None and odds_home is not None:
            form_ratio = safe_divide(home_ppg, home_ppg + away_ppg, 0.5)
            features["derived__agreement__form_vs_odds"] = (
                1.0 - abs(form_ratio - odds_home)
            )

    # ---------- Composites ----------

    def _compute_composites(self, features, pf):
        # Home/away strength = normalized weighted sum
        for side in ["home", "away"]:
            components = []
            ppg = pf.get(f"team__{side}__form__points_per_game__last5")
            if ppg is not None:
                components.append(ppg / 3.0)  # Normalize to 0-1
            xg = pf.get(f"team__{side}__form__avg_xg__last5")
            if xg is not None:
                components.append(min(xg / 3.0, 1.0))
            pos = pf.get(f"ctx__{side}__league_position")
            if pos is not None:
                components.append(max(0, 1.0 - pos / 20.0))
            squad_val = pf.get(f"player__{side}__squad_value_total")
            if squad_val is not None and squad_val > 0:
                components.append(min(math.log10(squad_val) / 10.0, 1.0))

            if components:
                features[f"derived__composite__{side}_strength"] = (
                    sum(components) / len(components)
                )

        # Match quality
        h_str = features.get("derived__composite__home_strength")
        a_str = features.get("derived__composite__away_strength")
        if h_str is not None and a_str is not None:
            features["derived__composite__match_quality"] = (
                (h_str + a_str) / 2.0
            )

        # Upset potential
        odds_home = pf.get("odds__implied_prob__home")
        odds_away = pf.get("odds__implied_prob__away")
        if odds_home is not None and odds_away is not None:
            fav_prob = max(odds_home, odds_away)
            features["derived__composite__upset_potential"] = 1.0 - fav_prob

        # Goals expectation
        h_gs = pf.get("team__home__form__goals_scored__last5")
        a_gs = pf.get("team__away__form__goals_scored__last5")
        h_lam = features.get("derived__poisson__home_lambda")
        a_lam = features.get("derived__poisson__away_lambda")

        components = []
        if h_gs is not None and a_gs is not None:
            components.append((h_gs + a_gs) / 5.0)
        if h_lam is not None and a_lam is not None:
            components.append(h_lam + a_lam)
        league_avg = pf.get("ctx__league__avg_goals")
        if league_avg is not None:
            components.append(league_avg)

        if components:
            features["derived__composite__goals_expectation"] = (
                sum(components) / len(components)
            )

    # ---------- Confidence ----------

    def _compute_confidence(self, features, pf):
        # Data completeness
        total = len(pf)
        non_null = sum(1 for v in pf.values() if v is not None)
        features["derived__confidence__data_completeness"] = (
            safe_divide(non_null, total) if total > 0 else 0.0
        )

        # Model consensus
        elo_home = features.get("derived__elo__home_expected")
        poisson_home = features.get("derived__poisson__home_win")
        odds_home = pf.get("odds__implied_prob__home")

        predictions = [
            p for p in [elo_home, poisson_home, odds_home] if p is not None
        ]
        if len(predictions) >= 2:
            std = float(np.std(predictions))
            features["derived__confidence__model_consensus"] = max(
                0, 1.0 - std * 3.0
            )
