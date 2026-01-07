"""
Test Suite for CMS Star Ratings Calculator
============================================
Verifies that the Python implementation exactly matches the SAS code logic.

These tests are structured to match each SAS file and macro:
- 0 - Data and Measure Standardization_2025Jul.sas
- 1 - First stage_Simple Average of Measure Scores_2025Jul.sas
- 2 - Second Stage_Weighted Average and Categorize Star_2025Jul.sas
- Star_Macros.sas
"""

import pytest
import numpy as np
from calculations import (
    StarRatingCalculator,
    MEASURE_GROUPS,
    MEASURES_TO_FLIP,
    STANDARD_WEIGHTS,
    GROUP_LABELS
)


class TestMeasureDefinitions:
    """Verify measure groups match SAS definitions"""
    
    def test_mortality_measures_match_sas(self):
        """
        SAS file 0, lines 179-181:
        ('MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF', 
         'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP')
        """
        expected = [
            'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
            'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP'
        ]
        assert set(MEASURE_GROUPS['mortality']) == set(expected)
        assert len(MEASURE_GROUPS['mortality']) == 7
    
    def test_safety_measures_match_sas(self):
        """
        SAS file 0, lines 202-203:
        ('COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4', 'HAI_5', 
         'HAI_6', 'PSI_90_SAFETY')
        """
        expected = [
            'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
            'HAI_5', 'HAI_6', 'PSI_90_SAFETY'
        ]
        assert set(MEASURE_GROUPS['safety']) == set(expected)
        assert len(MEASURE_GROUPS['safety']) == 8
    
    def test_readmission_measures_match_sas(self):
        """
        SAS file 0, lines 223-225:
        ('EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32', 'READM_30_CABG', 
         'READM_30_COPD', 'READM_30_HIP_KNEE', 'READM_30_HOSP_WIDE', 
         'OP_35_ADM', 'OP_35_ED', 'OP_36')
        """
        expected = [
            'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
            'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
            'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36'
        ]
        assert set(MEASURE_GROUPS['readmission']) == set(expected)
        assert len(MEASURE_GROUPS['readmission']) == 11
    
    def test_patient_experience_measures_match_sas(self):
        """
        SAS file 0, lines 245-247:
        ('H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 'H_COMP_3_STAR_RATING', 
         'H_COMP_5_STAR_RATING', 'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING', 
         'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING')
        """
        expected = [
            'H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 'H_COMP_3_STAR_RATING',
            'H_COMP_5_STAR_RATING', 'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING',
            'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING'
        ]
        assert set(MEASURE_GROUPS['patient_experience']) == set(expected)
        assert len(MEASURE_GROUPS['patient_experience']) == 8
    
    def test_process_measures_match_sas(self):
        """
        SAS file 0, lines 271-273:
        ('HCP_COVID_19', 'IMM_3', 'OP_10', 'OP_13', 'OP_18B', 'OP_22', 
         'OP_23', 'OP_29', 'OP_8', 'PC_01', 'SAFE_USE_OF_OPIOIDS', 'SEP_1')
        """
        expected = [
            'HCP_COVID_19', 'IMM_3', 'OP_10', 'OP_13', 'OP_18B',
            'OP_22', 'OP_23', 'OP_29', 'OP_8', 'PC_01',
            'SAFE_USE_OF_OPIOIDS', 'SEP_1'
        ]
        assert set(MEASURE_GROUPS['process']) == set(expected)
        assert len(MEASURE_GROUPS['process']) == 12
    
    def test_total_measure_count(self):
        """Total should be 46 measures as per SAS"""
        total = sum(len(m) for m in MEASURE_GROUPS.values())
        assert total == 46


class TestMeasureFlipping:
    """
    Verify measure re-direction matches SAS file 0, lines 319-357
    These measures have "lower is better" so they get negated
    """
    
    def test_all_mortality_measures_flipped(self):
        """SAS lines 319-325: All 7 mortality measures are negated"""
        mortality_measures = [
            'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
            'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP'
        ]
        for measure in mortality_measures:
            assert measure in MEASURES_TO_FLIP, f"{measure} should be flipped"
    
    def test_all_safety_measures_flipped(self):
        """SAS lines 327-334: All 8 safety measures are negated"""
        safety_measures = [
            'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
            'HAI_5', 'HAI_6', 'PSI_90_SAFETY'
        ]
        for measure in safety_measures:
            assert measure in MEASURES_TO_FLIP, f"{measure} should be flipped"
    
    def test_all_readmission_measures_flipped(self):
        """SAS lines 336-346: All 11 readmission measures are negated"""
        readmission_measures = [
            'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
            'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
            'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36'
        ]
        for measure in readmission_measures:
            assert measure in MEASURES_TO_FLIP, f"{measure} should be flipped"
    
    def test_patient_experience_not_flipped(self):
        """Patient experience measures are already 'higher is better'"""
        for measure in MEASURE_GROUPS['patient_experience']:
            assert measure not in MEASURES_TO_FLIP, f"{measure} should NOT be flipped"
    
    def test_process_measures_flipping(self):
        """SAS lines 348-357: Some process measures are negated"""
        should_be_flipped = ['OP_22', 'PC_01', 'OP_18B', 'OP_8', 'OP_10', 'OP_13', 'SAFE_USE_OF_OPIOIDS']
        should_not_be_flipped = ['HCP_COVID_19', 'IMM_3', 'OP_23', 'OP_29', 'SEP_1']
        
        for measure in should_be_flipped:
            assert measure in MEASURES_TO_FLIP, f"{measure} should be flipped"
        
        for measure in should_not_be_flipped:
            assert measure not in MEASURES_TO_FLIP, f"{measure} should NOT be flipped"


class TestStandardWeights:
    """
    Verify weights match SAS file 2, lines 63-67
    """
    
    def test_patient_experience_weight(self):
        """std_weight_PatientExperience = 0.22"""
        assert STANDARD_WEIGHTS['patient_experience'] == 0.22
    
    def test_readmission_weight(self):
        """std_weight_Readmission = 0.22"""
        assert STANDARD_WEIGHTS['readmission'] == 0.22
    
    def test_mortality_weight(self):
        """std_weight_Mortality = 0.22"""
        assert STANDARD_WEIGHTS['mortality'] == 0.22
    
    def test_safety_weight(self):
        """std_weight_safety = 0.22"""
        assert STANDARD_WEIGHTS['safety'] == 0.22
    
    def test_process_weight(self):
        """std_weight_Process = 0.12"""
        assert STANDARD_WEIGHTS['process'] == 0.12
    
    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0"""
        total = sum(STANDARD_WEIGHTS.values())
        assert abs(total - 1.0) < 0.0001


class TestGroupScoreCalculation:
    """
    Tests for %grp_score macro (Star_Macros.sas lines 28-63)
    
    SAS Logic:
    1. Count non-missing measures
    2. measure_wt = 1 / total_cnt
    3. avg = sum(measures) * measure_wt (i.e., simple average)
    """
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_group_score_simple_average(self):
        """
        Group score should be simple average of measures
        SAS: avg = sum(of &varlist.) * measure_wt where measure_wt = 1/total_cnt
        """
        # Using patient experience (not flipped) for simplicity
        measures = {
            'H_COMP_1_STAR_RATING': 4.0,
            'H_COMP_2_STAR_RATING': 3.0,
            'H_COMP_3_STAR_RATING': 5.0,
            # Others missing
        }
        
        score, count = self.calculator.calculate_group_score(measures, 'patient_experience')
        
        assert count == 3
        # Average = (4 + 3 + 5) / 3 = 4.0
        assert abs(score - 4.0) < 0.0001
    
    def test_group_score_empty_returns_none(self):
        """Empty group should return None score and 0 count"""
        measures = {}
        
        score, count = self.calculator.calculate_group_score(measures, 'mortality')
        
        assert score is None
        assert count == 0
    
    def test_group_score_count_matches_non_missing(self):
        """Count should match number of non-missing measures"""
        measures = {
            'MORT_30_AMI': 10.0,
            'MORT_30_HF': 12.0,
            'MORT_30_PN': None,  # Missing
            # Others implicitly missing
        }
        
        score, count = self.calculator.calculate_group_score(measures, 'mortality')
        
        assert count == 2


class TestEligibilityCheck:
    """
    Tests for %report macro (Star_Macros.sas lines 71-151)
    
    Eligibility criteria (SAS line 148):
    report_indicator = (MortSafe_Group_cnt >= 1) and (Total_measure_group_cnt >= 3)
    
    Where:
    - A group counts if it has >= 3 measures (SAS lines 139-142)
    - MortSafe_Group_cnt = mortality_has_3 + safety_has_3 (SAS line 146)
    - Total_measure_group_cnt = count of groups with >= 3 measures (SAS line 144)
    """
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_eligible_with_3_groups_including_mortality(self):
        """Hospital with 3+ groups including mortality should be eligible"""
        group_counts = {
            'mortality': 5,           # >= 3 ✓
            'safety': 2,              # < 3
            'readmission': 4,         # >= 3 ✓
            'patient_experience': 6,  # >= 3 ✓
            'process': 1              # < 3
        }
        
        eligible, reason = self.calculator.check_eligibility(group_counts)
        
        assert eligible is True
        assert "Eligible" in reason
    
    def test_eligible_with_3_groups_including_safety(self):
        """Hospital with 3+ groups including safety should be eligible"""
        group_counts = {
            'mortality': 1,           # < 3
            'safety': 5,              # >= 3 ✓
            'readmission': 4,         # >= 3 ✓
            'patient_experience': 6,  # >= 3 ✓
            'process': 1              # < 3
        }
        
        eligible, reason = self.calculator.check_eligibility(group_counts)
        
        assert eligible is True
    
    def test_ineligible_without_mortality_or_safety(self):
        """
        Hospital without mortality or safety having >= 3 should be ineligible
        Even if 3+ other groups have >= 3 measures
        """
        group_counts = {
            'mortality': 2,           # < 3
            'safety': 2,              # < 3
            'readmission': 8,         # >= 3 ✓
            'patient_experience': 7,  # >= 3 ✓
            'process': 10             # >= 3 ✓
        }
        
        eligible, reason = self.calculator.check_eligibility(group_counts)
        
        assert eligible is False
        assert "Mortality" in reason or "Safety" in reason
    
    def test_ineligible_with_fewer_than_3_groups(self):
        """Hospital with < 3 groups having >= 3 measures should be ineligible"""
        group_counts = {
            'mortality': 5,           # >= 3 ✓
            'safety': 5,              # >= 3 ✓
            'readmission': 1,         # < 3
            'patient_experience': 2,  # < 3
            'process': 0              # < 3
        }
        
        eligible, reason = self.calculator.check_eligibility(group_counts)
        
        assert eligible is False
        assert "2 groups" in reason  # Only 2 groups have >= 3
    
    def test_threshold_is_exactly_3(self):
        """Exactly 3 measures should count toward the threshold"""
        group_counts = {
            'mortality': 3,           # Exactly 3 ✓
            'safety': 3,              # Exactly 3 ✓
            'readmission': 3,         # Exactly 3 ✓
            'patient_experience': 2,  # < 3
            'process': 2              # < 3
        }
        
        eligible, reason = self.calculator.check_eligibility(group_counts)
        
        assert eligible is True


class TestSummaryScoreCalculation:
    """
    Tests for summary score calculation (SAS file 2, lines 40-93)
    
    Key logic:
    1. Identify missing groups (line 74)
    2. Redistribute weights: weight[k] = W[k] / (1 - sum_of_missing_weights) (line 83)
    3. Summary = sum(weight[k] * score[k]) (line 92)
    """
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_summary_score_all_groups_present(self):
        """When all groups present, use standard weights"""
        group_scores = {
            'mortality': 1.0,
            'safety': 1.0,
            'readmission': 1.0,
            'patient_experience': 1.0,
            'process': 1.0
        }
        
        summary, weights = self.calculator.calculate_summary_score(group_scores)
        
        # All scores = 1, so summary = sum of weights = 1.0
        assert abs(summary - 1.0) < 0.0001
        
        # Weights should equal standard weights
        assert abs(weights['mortality'] - 0.22) < 0.0001
        assert abs(weights['process'] - 0.12) < 0.0001
    
    def test_weight_redistribution_one_missing(self):
        """
        When one group missing, redistribute its weight
        SAS line 83: weight[k] = W[k] / (1 - sum_of_missing_weights)
        """
        group_scores = {
            'mortality': 1.0,
            'safety': 1.0,
            'readmission': None,  # Missing!
            'patient_experience': 1.0,
            'process': 1.0
        }
        
        summary, weights = self.calculator.calculate_summary_score(group_scores)
        
        # Readmission weight (0.22) gets redistributed
        # New denominator = 1 - 0.22 = 0.78
        expected_mort_weight = 0.22 / 0.78
        expected_process_weight = 0.12 / 0.78
        
        assert abs(weights['mortality'] - expected_mort_weight) < 0.0001
        assert abs(weights['process'] - expected_process_weight) < 0.0001
        assert weights['readmission'] is None
        
        # Summary = (0.22/0.78)*1 + (0.22/0.78)*1 + (0.22/0.78)*1 + (0.12/0.78)*1
        # = (0.22*3 + 0.12) / 0.78 = 0.78 / 0.78 = 1.0
        assert abs(summary - 1.0) < 0.0001
    
    def test_weight_redistribution_two_missing(self):
        """When two groups missing, redistribute both weights"""
        group_scores = {
            'mortality': 2.0,
            'safety': None,       # Missing
            'readmission': None,  # Missing
            'patient_experience': 2.0,
            'process': 2.0
        }
        
        summary, weights = self.calculator.calculate_summary_score(group_scores)
        
        # Missing weight sum = 0.22 + 0.22 = 0.44
        # New denominator = 1 - 0.44 = 0.56
        expected_mort_weight = 0.22 / 0.56
        expected_process_weight = 0.12 / 0.56
        
        assert abs(weights['mortality'] - expected_mort_weight) < 0.0001
        assert abs(weights['process'] - expected_process_weight) < 0.0001
        
        # Summary = 2 * (0.22 + 0.22 + 0.12) / 0.56 = 2 * 0.56 / 0.56 = 2.0
        assert abs(summary - 2.0) < 0.0001
    
    def test_weighted_average_calculation(self):
        """Verify weighted average formula with different scores"""
        group_scores = {
            'mortality': 0.5,
            'safety': 1.0,
            'readmission': -0.5,
            'patient_experience': 0.0,
            'process': 2.0
        }
        
        summary, weights = self.calculator.calculate_summary_score(group_scores)
        
        # Expected = 0.22*0.5 + 0.22*1.0 + 0.22*(-0.5) + 0.22*0.0 + 0.12*2.0
        #          = 0.11 + 0.22 - 0.11 + 0.0 + 0.24
        #          = 0.46
        expected = 0.22*0.5 + 0.22*1.0 + 0.22*(-0.5) + 0.22*0.0 + 0.12*2.0
        
        assert abs(summary - expected) < 0.0001


class TestPeerGroupAssignment:
    """
    Tests for peer group determination (SAS file 2, lines 121-127)
    
    Peer groups are based on number of groups with >= 3 measures:
    - cnt_grp = '1) # of groups=3' when Total = 3
    - cnt_grp = '2) # of groups=4' when Total = 4
    - cnt_grp = '3) # of groups=5' when Total = 5
    """
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_peer_group_3(self):
        """Hospital with 3 qualifying groups"""
        group_counts = {
            'mortality': 5,
            'safety': 5,
            'readmission': 5,
            'patient_experience': 2,
            'process': 1
        }
        
        peer = self.calculator.get_peer_group(group_counts)
        assert peer == 3
    
    def test_peer_group_4(self):
        """Hospital with 4 qualifying groups"""
        group_counts = {
            'mortality': 5,
            'safety': 5,
            'readmission': 5,
            'patient_experience': 5,
            'process': 1
        }
        
        peer = self.calculator.get_peer_group(group_counts)
        assert peer == 4
    
    def test_peer_group_5(self):
        """Hospital with all 5 qualifying groups"""
        group_counts = {
            'mortality': 5,
            'safety': 5,
            'readmission': 5,
            'patient_experience': 5,
            'process': 5
        }
        
        peer = self.calculator.get_peer_group(group_counts)
        assert peer == 5


class TestEndToEndCalculation:
    """Integration tests for complete calculation flow"""
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_eligible_hospital_gets_rating(self):
        """Complete flow for eligible hospital"""
        measures = {
            # 4 mortality measures (>= 3)
            'MORT_30_AMI': 10.0,
            'MORT_30_CABG': 3.0,
            'MORT_30_HF': 12.0,
            'MORT_30_PN': 14.0,
            # 4 safety measures (>= 3)
            'HAI_1': 0.8,
            'HAI_2': 0.9,
            'HAI_3': 1.0,
            'PSI_90_SAFETY': 0.95,
            # 4 patient experience measures (>= 3)
            'H_COMP_1_STAR_RATING': 4.0,
            'H_COMP_2_STAR_RATING': 4.0,
            'H_COMP_3_STAR_RATING': 3.0,
            'H_GLOB_STAR_RATING': 4.0,
        }
        
        result = self.calculator.calculate(measures)
        
        assert result.eligible is True
        assert result.star_rating is not None
        assert 1 <= result.star_rating <= 5
        assert result.summary_score is not None
        assert result.peer_group == 3  # 3 groups have >= 3 measures
    
    def test_ineligible_hospital_no_rating(self):
        """Hospital not meeting criteria gets no rating"""
        measures = {
            # Only 2 mortality measures
            'MORT_30_AMI': 10.0,
            'MORT_30_CABG': 3.0,
            # Only 2 patient experience
            'H_COMP_1_STAR_RATING': 4.0,
            'H_GLOB_STAR_RATING': 4.0,
        }
        
        result = self.calculator.calculate(measures)
        
        assert result.eligible is False
        assert result.star_rating is None
        assert result.summary_score is None
    
    def test_group_measure_counts_correct(self):
        """Verify measure counts per group are accurate"""
        measures = {
            'MORT_30_AMI': 10.0,
            'MORT_30_CABG': 3.0,
            'HAI_1': 0.8,
            'HAI_2': 0.9,
            'HAI_3': 1.0,
        }
        
        result = self.calculator.calculate(measures)
        
        assert result.group_measure_counts['mortality'] == 2
        assert result.group_measure_counts['safety'] == 3
        assert result.group_measure_counts['readmission'] == 0
        assert result.group_measure_counts['patient_experience'] == 0
        assert result.group_measure_counts['process'] == 0


class TestStarRatingBoundaries:
    """
    Test star rating assignment based on summary score
    
    The SAS uses K-means, but for single-hospital we use percentile approximation.
    These tests verify the boundary logic.
    """
    
    def setup_method(self):
        self.calculator = StarRatingCalculator()
    
    def test_very_low_score_gets_1_star(self):
        """Summary score < -0.84 should get 1 star"""
        star = self.calculator.estimate_star_rating(-1.5, peer_group=3)
        assert star == 1
    
    def test_low_score_gets_2_stars(self):
        """Summary score between -0.84 and -0.25 should get 2 stars"""
        star = self.calculator.estimate_star_rating(-0.5, peer_group=3)
        assert star == 2
    
    def test_average_score_gets_3_stars(self):
        """Summary score between -0.25 and 0.25 should get 3 stars"""
        star = self.calculator.estimate_star_rating(0.0, peer_group=3)
        assert star == 3
    
    def test_above_average_score_gets_4_stars(self):
        """Summary score between 0.25 and 0.84 should get 4 stars"""
        star = self.calculator.estimate_star_rating(0.5, peer_group=3)
        assert star == 4
    
    def test_high_score_gets_5_stars(self):
        """Summary score > 0.84 should get 5 stars"""
        star = self.calculator.estimate_star_rating(1.5, peer_group=3)
        assert star == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
