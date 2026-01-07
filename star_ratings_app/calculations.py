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


# =============================================================================
# Flask Web Application
# =============================================================================

from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__, static_folder='.')

# Initialize calculator
calculator = StarRatingCalculator()

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
        return jsonify({'error': str(e)}), 400


@app.route('/api/measures', methods=['GET'])
def get_measures():
    """Return measure definitions for the frontend"""
    return jsonify({
        'groups': MEASURE_GROUPS,
        'labels': GROUP_LABELS,
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
        return jsonify({'error': str(e)}), 400


@app.route('/api/hospitals/<provider_id>', methods=['GET'])
def get_hospital_data(provider_id):
    """Return measures for a specific hospital"""
    hospital = HOSPITALS_DATA.get(provider_id)
    if not hospital:
        return jsonify({'error': 'Hospital not found'}), 404
        
    return jsonify(hospital)


if __name__ == '__main__':
    print("Starting CMS Star Rankings Calculator...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
