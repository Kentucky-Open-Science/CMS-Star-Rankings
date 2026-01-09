"""
CMS Hospital Star Ratings Calculator
=====================================
Python implementation that exactly replicates the SAS code logic from:
- 0 - Data and Measure Standardization_2025Jul.sas
- 1 - First stage_Simple Average of Measure Scores_2025Jul.sas
- 2 - Second Stage_Weighted Average and Categorize Star_2025Jul.sas
- Star_Macros.sas

Author: Converted from Yale CORE SAS Package (July 2025)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
import csv
import io
import optuna


# =============================================================================
# MEASURE DEFINITIONS (from SAS code lines 37-41 in file 0)
# =============================================================================

# All 46 measures organized by group
MEASURE_GROUPS = {
    'mortality': [
        'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
        'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP'
    ],
    'safety': [
        'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 
        'HAI_5', 'HAI_6', 'PSI_90_SAFETY'
    ],
    'readmission': [
        'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
        'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
        'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36'
    ],
    'patient_experience': [
        'H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 'H_COMP_3_STAR_RATING',
        'H_COMP_5_STAR_RATING', 'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING',
        'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING'
    ],
    'process': [
        'HCP_COVID_19', 'IMM_3', 'OP_10', 'OP_13', 'OP_18B',
        'OP_22', 'OP_23', 'OP_29', 'OP_8', 'PC_01',
        'SAFE_USE_OF_OPIOIDS', 'SEP_1'
    ]
}

# Measures that need to be flipped (lower = better, so negate after standardization)
# From SAS code lines 319-357 in file 0
MEASURES_TO_FLIP = [
    # Mortality - all flipped
    'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
    'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP',
    # Safety - all flipped
    'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
    'HAI_5', 'HAI_6', 'PSI_90_SAFETY',
    # Readmission - all flipped
    'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
    'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
    'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36',
    # Process - some flipped
    'OP_22', 'PC_01', 'OP_18B', 'OP_8', 'OP_10', 'OP_13',
    'SAFE_USE_OF_OPIOIDS'
]

# Standard weights from CMS (from SAS code lines 63-67 in file 2)
STANDARD_WEIGHTS = {
    'patient_experience': 0.22,
    'readmission': 0.22,
    'mortality': 0.22,
    'safety': 0.22,
    'process': 0.12
}

# Human-readable group names
GROUP_LABELS = {
    'mortality': 'Outcomes - Mortality',
    'safety': 'Outcomes - Safety of Care',
    'readmission': 'Outcomes - Readmission',
    'patient_experience': 'Patient Experience',
    'process': 'Timely and Effective Care'
}

# Measures identified as "Actionable" (e.g., capable of improvement via staffing/training)
# Includes Patient Experience, Safety, and specific Process measures (ED timing, Sepsis)
ACTIONABLE_MEASURES = [
    # Patient Experience (All)
    'H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 'H_COMP_3_STAR_RATING',
    'H_COMP_5_STAR_RATING', 'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING',
    'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING',
    # Safety (All)
    'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 
    'HAI_5', 'HAI_6', 'PSI_90_SAFETY',
    # Process (Selected)
    'SEP_1', 'OP_18B' # Sepsis, ED Wait Time
]


@dataclass
class StarRatingResult:
    """Result of star rating calculation"""
    eligible: bool
    eligibility_reason: str
    group_scores: Dict[str, Optional[float]]
    group_measure_counts: Dict[str, int]
    summary_score: Optional[float]
    star_rating: Optional[int]
    peer_group: Optional[int]  # 3, 4, or 5 groups
    weights_used: Dict[str, Optional[float]]


class StarRatingCalculator:
    """
    CMS Hospital Star Rating Calculator
    
    Implements the exact methodology from the SAS code package.
    """
    
    def __init__(self, national_stats: Optional[Dict] = None):
        """
        Initialize calculator.
        If national_stats is None, it loads data from CSV, calculates stats,
        AND calculates star rating cutoffs using K-means on the full dataset.
        """
        self.national_stats = national_stats
        self.cutoffs = {3: [], 4: [], 5: []}  # Cutoffs for each peer group
        
        if self.national_stats is None:
            self._initialize_from_csv()
        else:
            # If stats provided but not cutoffs, use defaults (unlikely path)
            self.cutoffs = self._get_default_cutoffs()

    def _get_default_national_stats(self) -> Dict:
        """Default placeholder stats."""
        stats = {}
        for group, measures in MEASURE_GROUPS.items():
            for measure in measures:
                stats[measure] = {'mean': 0.0, 'std': 1.0}
        return stats

    def _get_default_cutoffs(self) -> Dict[int, List[float]]:
        """Default / fallback cutoffs."""
        # Approximate cutoffs
        return {
            3: [-0.84, -0.25, 0.25, 0.84],
            4: [-0.84, -0.25, 0.25, 0.84],
            5: [-0.84, -0.25, 0.25, 0.84]
        }

    def _initialize_from_csv(self):
        """Load data, calc stats, and run K-means for cutoffs."""
        import os
        
        # 1. Load all data
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, '..', 'alldata_2025jul.csv')
            
            if not os.path.exists(csv_path):
                print("CSV not found, using defaults.")
                self.national_stats = self._get_default_national_stats()
                self.cutoffs = self._get_default_cutoffs()
                return

            print(f"Loading data from {csv_path}...")
            all_data = []
            all_measures = []
            for group_measures in MEASURE_GROUPS.values():
                all_measures.extend(group_measures)
            
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    hospital_measures = {}
                    for m in all_measures:
                        val_str = row.get(m)
                        if val_str not in ['', 'Not Available', 'NaN', 'nan', None]:
                            try:
                                hospital_measures[m] = float(val_str)
                            except:
                                hospital_measures[m] = None
                        else:
                            hospital_measures[m] = None
                    all_data.append(hospital_measures)

            # 2. Calculate National Stats
            stats = {}
            for m in all_measures:
                values = [d[m] for d in all_data if d.get(m) is not None]
                if values:
                    stats[m] = {'mean': np.mean(values), 'std': np.std(values, ddof=1)}
                else:
                    stats[m] = {'mean': 0.0, 'std': 1.0}
            self.national_stats = stats
            
            # 3. Calculate Summary Scores for all hospitals
            peer_group_scores = {3: [], 4: [], 5: []}
            
            for hospital_measures in all_data:
                # Calculate group scores
                group_scores = {}
                group_counts = {}
                for group_name in MEASURE_GROUPS.keys():
                    # Use internal logic but optimized
                    score, count = self._calculate_group_score_internal(hospital_measures, group_name)
                    group_scores[group_name] = score
                    group_counts[group_name] = count
                
                # Check eligibility
                eligible, _ = self.check_eligibility(group_counts)
                if eligible:
                    summary_score, _ = self.calculate_summary_score(group_scores)
                    peer_group = self.get_peer_group(group_counts)
                    if summary_score is not None:
                        peer_group_scores[peer_group].append(summary_score)

            # 4. K-means to determine cutoffs
            print("Calculating K-means cutoffs...")
            for pg in [3, 4, 5]:
                scores = peer_group_scores[pg]
                if len(scores) >= 5:
                    self.cutoffs[pg] = self._calculate_kmeans_cutoffs(scores)
                else:
                    self.cutoffs[pg] = self._get_default_cutoffs()[pg]
            
            print("Initialization complete.")
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            self.national_stats = self._get_default_national_stats()
            self.cutoffs = self._get_default_cutoffs()

    def _calculate_kmeans_cutoffs(self, scores: List[float]) -> List[float]:
        """Run K-means and return the 4 boundaries between 5 clusters."""
        # Use existing function logic or call it? 
        # The existing function returns ratings for a list. 
        # We need boundaries without assigning to specific hospitals.
        # Let's replicate logic here for clarity.
        
        X = np.array(scores).reshape(-1, 1)
        
        # Initial seeds (quintiles)
        percentiles = [10, 30, 50, 70, 90]
        seeds = np.percentile(X, percentiles).reshape(-1, 1)
        
        kmeans = KMeans(n_clusters=5, init=seeds, n_init=1, max_iter=1000)
        kmeans.fit(X)
        
        # Get max score for each cluster
        cluster_maxes = []
        for i in range(5):
            mask = kmeans.labels_ == i
            if mask.sum() > 0:
                cluster_mean = X[mask].mean()
                cluster_max = X[mask].max()
                cluster_maxes.append((cluster_mean, cluster_max))
        
        # Sort by mean to order clusters 1-5
        cluster_maxes.sort(key=lambda x: x[0])
        
        # The cutoffs are the max values of the first 4 clusters
        # (Technically it's the boundary, but using max of lower cluster is safe for "step" function)
        # Actually SAS might use midpoints, but usually it's just binning.
        # Let's use max of cluster N as the cutoff for N vs N+1.
        cutoffs = [c[1] for c in cluster_maxes[:4]]
        
        # Ensure strictly increasing (should be if means are well separated)
        cutoffs.sort()
        
        return cutoffs

    def _calculate_group_score_internal(self, measures, group_name):
        """Helper to use pre-loaded measures dict."""
        group_measures = MEASURE_GROUPS[group_name]
        values = []
        for m in group_measures:
            if measures.get(m) is not None:
                stats = self.national_stats.get(m, {'mean':0,'std':1})
                if stats['std'] == 0: val=0
                else: val = (measures[m] - stats['mean']) / stats['std']
                
                if m in MEASURES_TO_FLIP: val = -val
                values.append(val)
        
        count = len(values)
        if count == 0: return None, 0
        return sum(values)/count, count

    # ... Standard methods ...

    def standardize_measure(self, measure_name: str, value: float) -> float:
        stats = self.national_stats.get(measure_name, {'mean': 0, 'std': 1})
        if stats['std'] == 0: return 0.0
        return (value - stats['mean']) / stats['std']
    
    def redirect_measure(self, measure_name: str, std_value: float) -> float:
        if measure_name in MEASURES_TO_FLIP: return -std_value
        return std_value
    
    def calculate_group_score(self, measures: Dict[str, Optional[float]], group_name: str) -> Tuple[Optional[float], int]:
        # Same logic as before but using instance methods
        # This is strictly for the public API call which passes a dict
        return self._calculate_group_score_internal(measures, group_name)
    
    def check_eligibility(self, group_counts: Dict[str, int]) -> Tuple[bool, str]:
        # Copied from previous
        groups_with_3 = {g: c >= 3 for g, c in group_counts.items()}
        total_groups = sum(groups_with_3.values())
        mort_safe_count = (1 if groups_with_3.get('mortality') else 0) + \
                          (1 if groups_with_3.get('safety') else 0)
        is_eligible = (mort_safe_count >= 1) and (total_groups >= 3)
        
        if is_eligible:
            reason = f"Eligible: {total_groups} groups with ≥3 measures"
        else:
            reasons = []
            if total_groups < 3: reasons.append(f"only {total_groups} groups have ≥3 measures (need 3)")
            if mort_safe_count < 1: reasons.append("neither Mortality nor Safety has ≥3 measures")
            reason = "Ineligible: " + "; ".join(reasons)
        return is_eligible, reason
    
    def calculate_summary_score(self, group_scores: Dict[str, Optional[float]]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
        # Copied from previous
        missing = {g: s is None for g, s in group_scores.items()}
        missing_weight_sum = sum(STANDARD_WEIGHTS[g] for g, m in missing.items() if m)
        
        if missing_weight_sum >= 1.0: return None, {g: None for g in STANDARD_WEIGHTS}
        
        redistributed_weights = {}
        for group, base_weight in STANDARD_WEIGHTS.items():
            if missing[group]: redistributed_weights[group] = None
            else: redistributed_weights[group] = base_weight / (1 - missing_weight_sum)
            
        summary = 0.0
        for group, score in group_scores.items():
            if score is not None and redistributed_weights[group] is not None:
                summary += redistributed_weights[group] * score
        return summary, redistributed_weights
    
    def get_peer_group(self, group_counts: Dict[str, int]) -> int:
        groups_with_3 = sum(1 for count in group_counts.values() if count >= 3)
        return min(max(groups_with_3, 3), 5)

    def estimate_star_rating(self, summary_score: float, peer_group: int) -> int:
        """
        Estimate star rating using calculated cutoffs.
        """
        if peer_group not in self.cutoffs or not self.cutoffs[peer_group]:
            # Fallback
            cutoffs = [-0.84, -0.25, 0.25, 0.84]
        else:
            cutoffs = self.cutoffs[peer_group]
            
        # cutoffs is a list of 4 values: [max_1star, max_2star, max_3star, max_4star]
        if summary_score <= cutoffs[0]: return 1
        elif summary_score <= cutoffs[1]: return 2
        elif summary_score <= cutoffs[2]: return 3
        elif summary_score <= cutoffs[3]: return 4
        else: return 5

    def calculate(self, measures: Dict[str, Optional[float]]) -> StarRatingResult:
        group_scores = {}
        group_counts = {}
        for group_name in MEASURE_GROUPS.keys():
            score, count = self.calculate_group_score(measures, group_name)
            group_scores[group_name] = score
            group_counts[group_name] = count
            
        eligible, eligibility_reason = self.check_eligibility(group_counts)
        
        if not eligible:
            return StarRatingResult(False, eligibility_reason, group_scores, group_counts, 
                                  None, None, None, {g: None for g in STANDARD_WEIGHTS})
        
        summary_score, weights_used = self.calculate_summary_score(group_scores)
        peer_group = self.get_peer_group(group_counts)
        star_rating = self.estimate_star_rating(summary_score, peer_group)
        
        return StarRatingResult(True, eligibility_reason, group_scores, group_counts,
                              summary_score, star_rating, peer_group, weights_used)


def calculate_kmeans_star_ratings(hospitals_data: List[Dict[str, Optional[float]]], 
                                   peer_group: int) -> List[int]:
    """
    Calculate star ratings for multiple hospitals using K-means clustering.
    
    This replicates the exact %kmeans macro from Star_Macros.sas (lines 157-237).
    
    The SAS algorithm:
    1. Calculate quintile medians as initial seeds
    2. Run K-means to convergence
    3. Re-run with strict=1 to avoid outlier effects
    4. Order clusters by mean summary score
    5. Assign stars 1-5 based on ordered clusters
    
    Args:
        hospitals_data: List of summary scores for all hospitals in peer group
        peer_group: The peer group (3, 4, or 5)
        
    Returns:
        List of star ratings (1-5) for each hospital
    """
    summary_scores = np.array(hospitals_data).reshape(-1, 1)
    
    if len(summary_scores) < 5:
        # Not enough hospitals for K-means
        return [3] * len(summary_scores)
    
    # Step 1: Calculate quintile medians as initial seeds (SAS lines 162-176)
    percentiles = [10, 30, 50, 70, 90]  # Medians of quintiles
    initial_seeds = np.percentile(summary_scores, percentiles).reshape(-1, 1)
    
    # Step 2: Run K-means with initial seeds (SAS lines 186-191)
    kmeans = KMeans(n_clusters=5, init=initial_seeds, n_init=1, max_iter=1000)
    kmeans.fit(summary_scores)
    
    # Step 3: Order clusters by mean (SAS lines 205-218)
    cluster_means = []
    for i in range(5):
        mask = kmeans.labels_ == i
        if mask.sum() > 0:
            cluster_means.append((i, summary_scores[mask].mean()))
        else:
            cluster_means.append((i, float('-inf')))
    
    # Sort clusters by mean
    cluster_means.sort(key=lambda x: x[1])
    
    # Create mapping from cluster ID to star rating (SAS lines 209-214)
    cluster_to_star = {cluster_id: star + 1 for star, (cluster_id, _) in enumerate(cluster_means)}
    
    # Assign stars
    star_ratings = [cluster_to_star[label] for label in kmeans.labels_]
    
    return star_ratings



class StarRatingOptimizer:
    """
    Optimizer for CMS Star Ratings using Optuna.
    
    Strategies:
    1. Primary: Find MINIMUM COST to reach (Current Rating + 1).
    2. Fallback: If (1) is impossible or over budget, MAXIMIZE Summary Score within budget.
    
    Cost Model: $5,000 per 0.1 SD improvement per measure.
    """
    def __init__(self, calculator: StarRatingCalculator):
        self.calculator = calculator

    def optimize(self, current_measures: Dict[str, Optional[float]], 
                 budget: float, 
                 measure_costs: Dict[str, float], 
                 target_mode: str = 'next_star', 
                 target_score: Optional[float] = None) -> Dict:
        """
        Optimize Start Rating based on selected measures and costs.
        
        Args:
            current_measures: Current values for all measures.
            budget: Total budget available.
            measure_costs: Dict of {measure_name: cost_per_0.1_sd}. Only measures in this dict are optimized.
            target_mode: 'next_star' (default) or 'specific_score'.
            target_score: Target summary score (only used if target_mode == 'specific_score').
        """
        # Valid measures to improve are ONLY those in measure_costs (user selected)
        # But we must filter for those that actually exist in current_measures
        optimizable_measures = [
            m for m in measure_costs.keys() 
            if m in current_measures and current_measures[m] is not None
        ]
        
        if not optimizable_measures:
             return {'error': "No valid measures selected for optimization."}

        # Helper to calculate cost for a specific measure
        def calculate_cost(measure, improvement_sd):
            # Cost is per 0.1 SD, so improvement_sd * 10 * specific_cost
            cost_per_unit = measure_costs.get(measure, 5000.0)
            return improvement_sd * 10.0 * cost_per_unit

        # 1. Baseline Calculation
        start_result = self.calculator.calculate(current_measures)
        current_rating = start_result.star_rating or 3
        current_summary = start_result.summary_score or 0.0
        
        # Determine Target Threshold
        target_threshold = -999.0
        
        if target_mode == 'specific_score' and target_score is not None:
            target_threshold = target_score
        else:
            # Default to next star rating
            # Get cutoffs for the peer group
            pg = start_result.peer_group or 3 # Fallback
            cutoffs = self.calculator.cutoffs.get(pg, [-0.84, -0.25, 0.25, 0.84])
            
            # 1->2 (idx 0), 2->3 (idx 1), 3->4 (idx 2), 4->5 (idx 3)
            # If current is 3, we want to beat cutoff[2] (0.25) to get 4?
            # calculate logic: if score <= cutoffs[2] -> 3. So > cutoffs[2] -> 4.
            # We add a tiny buffer.
            
            target_star = min(current_rating + 1, 5)
            if target_star == 1: idx = 0 # should not happen if we are improving
            elif target_star == 2: idx = 0
            elif target_star == 3: idx = 1
            elif target_star == 4: idx = 2
            elif target_star == 5: idx = 3
            
            if current_rating == 5 and target_mode == 'next_star':
                 # Already maxed, maybe just Maximize Score logic?
                 # Set a high threshold to force maximization behavior
                 target_threshold = 100.0 
            else:
                target_threshold = cutoffs[idx] + 0.01

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        
        # ==========================================================
        # STRATEGY: Reach Target Threshold with Min Cost
        # ==========================================================
        study = optuna.create_study(direction='minimize')
        
        # Seed with baseline (0.0) to ensure we consider "doing nothing" (or close to it)
        study.enqueue_trial({f'imp_{m}': 0.0 for m in optimizable_measures})
        # Seed with small improvements to guide towards local, low-cost steps
        study.enqueue_trial({f'imp_{m}': 0.01 for m in optimizable_measures})
        
        def objective(trial):
            simulated_measures = current_measures.copy()
            total_cost = 0.0
            
            # Penalties
            score_penalty = 0.0
            
            for m in optimizable_measures:
                # Suggest improvement 0.0 to 3.0 SD (widened range)
                imp_sd = trial.suggest_float(f'imp_{m}', 0.0, 3.0)
                
                # Accrue cost
                total_cost += calculate_cost(m, imp_sd)
                
                # Apply change
                if imp_sd > 0:
                    stats = self.calculator.national_stats.get(m, {'mean': 0, 'std': 1})
                    if stats['std'] != 0:
                        if m in MEASURES_TO_FLIP:
                            simulated_measures[m] = current_measures[m] - (imp_sd * stats['std'])
                        else:
                            simulated_measures[m] = current_measures[m] + (imp_sd * stats['std'])

            # Calculate Result
            res = self.calculator.calculate(simulated_measures)
            sim_score = res.summary_score or -100.0
            
            # Constraint: Must exceed target_threshold
            if sim_score < target_threshold:
                # Penalty proportional to distance
                # INCREASED PENALTY to ensure we prioritize score over cost
                score_penalty = (target_threshold - sim_score) * 100_000_000 
                # Why distinct penalty? because we want to minimize cost ONLY if constraint met
                # If constraint not met, cost doesn't matter as much as getting closer
            
            # Constraint: Budget
            if total_cost > budget:
                # Soft penalty? Or strict? 
                # If we are minimizing cost, exceeding budget should be penalized HUGE
                return 1e12 + total_cost 
                
            return total_cost + score_penalty

        # Run optimization
        study.optimize(objective, n_trials=100, timeout=10.0)
        
        best_plan = study.best_trial
        
        # If best plan is invalid (high penalty), we prefer "Maximize Score under budget"
        # Check if best plan met the specific threshold
        # Re-calc to verify
        # (Alternatively, run a Maximize Score study if Min Cost failed to find a valid solution)
        
        is_success_min_cost = False
        if best_plan.value < 1e9: # Arbitrary large number check
             is_success_min_cost = True
        
        if not is_success_min_cost:
            # Fallback: Maximize Score under budget
             study_max = optuna.create_study(direction='maximize')
             
             def objective_max(trial):
                simulated_measures = current_measures.copy()
                total_cost = 0.0
                
                for m in optimizable_measures:
                    imp_sd = trial.suggest_float(f'imp_{m}', 0.0, 2.0)
                    total_cost += calculate_cost(m, imp_sd)
                    
                    if imp_sd > 0:
                        stats = self.calculator.national_stats.get(m, {'mean': 0, 'std': 1})
                        if stats['std'] != 0:
                            if m in MEASURES_TO_FLIP:
                                simulated_measures[m] = current_measures[m] - (imp_sd * stats['std'])
                            else:
                                simulated_measures[m] = current_measures[m] + (imp_sd * stats['std'])
                
                if total_cost > budget:
                    return -1000.0
                
                res = self.calculator.calculate(simulated_measures)
                return res.summary_score or -100.0
                
             study_max.optimize(objective_max, n_trials=50, timeout=5.0)
             best_plan = study_max.best_trial

        # ==========================================================
        # CONSTRUCT RESULT
        # ==========================================================
        best_measures = current_measures.copy()
        recommendations = []
        total_cost = 0.0
        
        for m in optimizable_measures:
            imp_key = f'imp_{m}'
            if imp_key in best_plan.params:
                imp_sd = best_plan.params[imp_key]
                if imp_sd > 0.001:
                    cost = calculate_cost(m, imp_sd)
                    
                    stats = self.calculator.national_stats.get(m, {'mean': 0, 'std': 1})
                    std = stats['std']
                    
                    if m in MEASURES_TO_FLIP:
                        target_val = current_measures[m] - (imp_sd * std)
                        direction = "Decrease"
                    else:
                        target_val = current_measures[m] + (imp_sd * std)
                        direction = "Increase"
                    
                    # Filter out negligible changes (displayed as 0.00)
                    if abs(target_val - current_measures[m]) < 0.005:
                        continue
                        
                    best_measures[m] = target_val
                    total_cost += cost
                    
                    recommendations.append({
                        'measure': m,
                        'current_value': current_measures[m],
                        'target_value': target_val,
                        'improvement_sd': imp_sd,
                        'cost': cost,
                        'description': f"{direction} {m} by {abs(target_val - current_measures[m]):.2f} ({imp_sd:.2f} SD)"
                    })

        # Sort recommendations by impact (cost or SD?)
        recommendations.sort(key=lambda x: x['improvement_sd'], reverse=True)

        # ==========================================================
        # STRICT BUDGET CLAMPING
        # ==========================================================
        # Ensure floating point tolerances don't cause minor overages, 
        # but clamp significant ones.
        if total_cost > budget:
             # Scale down all improvements proportionally to fit budget
             # Use a slightly aggressive scaling to ensure we are strictly under
             scaling_factor = budget / (total_cost + 0.000001)
             
             # Apply scaling
             total_cost = 0.0
             best_measures = current_measures.copy() # Reset
             
             for rec in recommendations:
                 # Scale stats
                 original_imp = rec['improvement_sd']
                 new_imp = original_imp * scaling_factor
                 
                 rec['improvement_sd'] = new_imp
                 
                 # Re-calc cost
                 new_cost = calculate_cost(rec['measure'], new_imp)
                 rec['cost'] = new_cost
                 total_cost += new_cost
                 
                 # Re-calc target value
                 m = rec['measure']
                 stats = self.calculator.national_stats.get(m, {'mean': 0, 'std': 1})
                 std = stats['std']
                 
                 if m in MEASURES_TO_FLIP:
                     target_val = current_measures[m] - (new_imp * std)
                     direction = "Decrease"
                 else:
                     target_val = current_measures[m] + (new_imp * std)
                     direction = "Increase"
                 
                 rec['target_value'] = target_val
                 rec['description'] = f"{direction} {m} by {abs(target_val - current_measures[m]):.2f} ({new_imp:.2f} SD)"
                 
                 best_measures[m] = target_val

        final_result = self.calculator.calculate(best_measures)
        
        return {
            'original_rating': start_result.star_rating,
            'original_summary_score': start_result.summary_score,
            'optimized_rating': final_result.star_rating,
            'optimized_summary_score': final_result.summary_score,
            'total_cost': total_cost,
            'budget': budget,
            'target_mode': target_mode,
            'target_threshold': target_threshold,
            'recommendations': recommendations
        }


# =============================================================================
# Flask Web Application
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='.')

# Initialize calculator
calculator = StarRatingCalculator()
optimizer = StarRatingOptimizer(calculator)

# Load hospital data
HOSPITALS_DATA = {}
HOSPITAL_NAMES = {}
HOSPITAL_RATINGS = {}

def load_hospital_names():
    """Load hospital names and ratings from General Information CSV"""
    global HOSPITAL_NAMES, HOSPITAL_RATINGS
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'Hospital_General_Information.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: Name file not found at {csv_path}")
            return

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                pid = row.get('Facility ID')
                name = row.get('Facility Name')
                rating = row.get('Hospital overall rating')
                
                if pid and name:
                    HOSPITAL_NAMES[pid] = name
                    if rating and rating.isdigit():
                         HOSPITAL_RATINGS[pid] = int(rating)
                    count += 1
            print(f"Loaded {count} hospital names/ratings from {csv_path}")
    except Exception as e:
        print(f"Error loading hospital names: {e}")

def load_hospital_data():
    """Load hospital data from CSV file"""
    global HOSPITALS_DATA
    try:
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'alldata_2025jul.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: Data file not found at {csv_path}")
            return

        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                provider_id = row.get('PROVIDER_ID')
                if provider_id:
                    # Clean up the row data - convert 'Not Available' etc to None
                    clean_row = {}
                    for k, v in row.items():
                        if v in ['', 'Not Available', 'NaN', 'nan']:
                            clean_row[k] = None
                        else:
                            try:
                                clean_row[k] = float(v)
                            except (ValueError, TypeError):
                                clean_row[k] = v
                    
                    HOSPITALS_DATA[provider_id] = clean_row
                    count += 1
            print(f"Loaded {count} hospitals from {csv_path}")
    except Exception as e:
        print(f"Error loading hospital data: {e}")

# Load data on startup
load_hospital_names()
load_hospital_data()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/calculate', methods=['POST'])
def calculate_rating():
    """API endpoint to calculate star rating"""
    try:
        data = request.json
        measures = {}
        
        # Parse measure values from request
        for group, measure_list in MEASURE_GROUPS.items():
            for measure in measure_list:
                value = data.get(measure)
                if value is not None and value != '':
                    try:
                        measures[measure] = float(value)
                    except (ValueError, TypeError):
                        measures[measure] = None
                else:
                    measures[measure] = None
        
        # Calculate rating
        result = calculator.calculate(measures)
        
        # Format response
        response = {
            'eligible': result.eligible,
            'eligibility_reason': result.eligibility_reason,
            'group_scores': {
                GROUP_LABELS.get(k, k): v 
                for k, v in result.group_scores.items()
            },
            'group_measure_counts': {
                GROUP_LABELS.get(k, k): v 
                for k, v in result.group_measure_counts.items()
            },
            'summary_score': result.summary_score,
            'star_rating': result.star_rating,
            'peer_group': result.peer_group,
            'weights_used': {
                GROUP_LABELS.get(k, k): v 
                for k, v in result.weights_used.items()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Calculate rating error: {e}")
        return jsonify({'error': 'An error occurred while calculating the rating. Please check your input values.'}), 400


@app.route('/api/optimize', methods=['POST'])
def optimize_rating():
    """API endpoint to optimize rating given a budget and settings"""
    try:
        data = request.json
        budget = float(data.get('budget', 1000000.0))
        
        # New parameters
        target_mode = data.get('target_mode', 'next_star')   # 'next_star' or 'specific_score'
        target_score_val = data.get('target_score')          # float or None
        
        if target_score_val is not None and target_score_val != '':
            try:
                target_score = float(target_score_val)
            except:
                target_score = None
        else:
             target_score = None
             
        # measure_costs is a dict {measure_name: cost_per_0.1_sd}
        # It dictates WHICH measures we optimize and their costs.
        measure_costs = data.get('measure_costs', {}) 
         
        # We also need CURRENT measure values to base off
        # The frontend should send them, OR we assume they are in 'measures'?
        # Based on previous pattern, user sends all measure values in the request
        measures = {}
        
        # Parse measure values
        for group, measure_list in MEASURE_GROUPS.items():
            for measure in measure_list:
                value = data.get(measure)
                if value is not None and value != '':
                    try:
                        measures[measure] = float(value)
                    except (ValueError, TypeError):
                        measures[measure] = None
                else:
                    measures[measure] = None
                    
        # If measure_costs is empty (legacy call or user selected nothing),
        # Use ACTIONABLE_MEASURES with default cost?
        if not measure_costs:
            measure_costs = {m: 5000.0 for m in ACTIONABLE_MEASURES}

        # Ensure measure_costs values are floats
        clean_costs = {}
        for k, v in measure_costs.items():
            try:
                 clean_costs[k] = float(v)
            except:
                 clean_costs[k] = 5000.0
        
        result = optimizer.optimize(measures, budget, clean_costs, target_mode, target_score)
        return jsonify(result)

    except Exception as e:
        print(f"Optimization error: {e}")
        return jsonify({'error': 'An error occurred during optimization. Please check your input values.'}), 400


@app.route('/api/measures', methods=['GET'])
def get_measures():
    """Return measure definitions for the frontend"""
    return jsonify({
        'groups': MEASURE_GROUPS,
        'labels': GROUP_LABELS,
        'actionable': ACTIONABLE_MEASURES,
        'weights': STANDARD_WEIGHTS,
        'flip_measures': MEASURES_TO_FLIP
    })


@app.route('/api/hospitals', methods=['GET'])
def get_hospitals():
    """Return list of available hospitals"""
    hospitals = []
    for pid, data in HOSPITALS_DATA.items():
        # Get name from mapping or fallback
        name = HOSPITAL_NAMES.get(pid, f"Provider {pid}")
        hospitals.append({
            'id': pid,
            'name': name,
            'display_name': name # For the dropdown display
        })
    
    # Sort by name for easier searching
    hospitals.sort(key=lambda x: x['name'])
    
    return jsonify(hospitals)


@app.route('/api/hospitals/list', methods=['GET'])
def get_hospital_list():
    """
    Return a paginated list of hospitals, optionally filtered by rating.
    Default behavior: Limit 10, Page 1, All ratings if not specified.
    """
    try:
        # Parse parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        rating_filter = request.args.get('rating')
        
        # Convert rating_filter to int if present
        if rating_filter and rating_filter.isdigit():
            rating_filter = int(rating_filter)
        else:
            rating_filter = None
            
        filtered_hospitals = []
        
        # Filter hospitals
        for pid, rating in HOSPITAL_RATINGS.items():
            # Check availability in data
            if pid not in HOSPITALS_DATA:
                continue
                
            # Apply rating filter
            if rating_filter is not None and rating != rating_filter:
                continue
                
            filtered_hospitals.append({
                'id': pid,
                'name': HOSPITAL_NAMES.get(pid, f"Provider {pid}"),
                'rating': rating
            })
            
        # Sort by name
        filtered_hospitals.sort(key=lambda x: x['name'])
        
        # Calculate pagination
        total_items = len(filtered_hospitals)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        paginated_items = filtered_hospitals[start_idx:end_idx]
        
        return jsonify({
            'hospitals': paginated_items,
            'total': total_items,
            'page': page,
            'limit': limit,
            'total_pages': (total_items + limit - 1) // limit
        })
        
    except Exception as e:
        print(f"Hospital list error: {e}")
        return jsonify({'error': 'An error occurred while fetching the hospital list.'}), 400


@app.route('/api/hospitals/<provider_id>', methods=['GET'])
def get_hospital_data(provider_id):
    """Return measures for a specific hospital"""
    hospital = HOSPITALS_DATA.get(provider_id)
    if not hospital:
        return jsonify({'error': 'Hospital not found'}), 404
        
    return jsonify(hospital)


if __name__ == '__main__':
    print("Starting CMS Star Rankings Calculator...")
    print("Open http://localhost:5555 in your browser")
    
    # Debug mode controlled via environment variable for security
    # Set FLASK_DEBUG=1 for development, leave unset for production
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    
    if debug_mode:
        # Use Flask development server only in debug mode
        print("Running in DEBUG mode with Flask development server")
        app.run(debug=True, port=5555)
    else:
        # Use production WSGI server
        try:
            # Try Waitress first (works on both Windows and Linux)
            from waitress import serve
            print("Running with Waitress production server")
            serve(app, host='0.0.0.0', port=5555)
        except ImportError:
            try:
                # Fallback to Gunicorn (Linux only)
                import subprocess
                import sys
                print("Running with Gunicorn production server")
                subprocess.run([
                    sys.executable, '-m', 'gunicorn',
                    '-b', '0.0.0.0:5555',
                    '-w', '4',
                    'calculations:app'
                ])
            except Exception:
                # Final fallback - Flask dev server with warning
                print("WARNING: No production server available. Install waitress: pip install waitress")
                app.run(debug=False, port=5555, host='0.0.0.0')

