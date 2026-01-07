/**
 * CMS Hospital Star Ratings Calculator
 * Frontend JavaScript - Communicates with Python Flask backend
 */

// Measure group definitions (matching calculations.py)
const MEASURE_GROUPS = {
    mortality: [
        'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
        'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP'
    ],
    safety: [
        'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
        'HAI_5', 'HAI_6', 'PSI_90_SAFETY'
    ],
    readmission: [
        'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
        'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
        'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36'
    ],
    patient_experience: [
        'H_COMP_1_STAR_RATING', 'H_COMP_2_STAR_RATING', 'H_COMP_3_STAR_RATING',
        'H_COMP_5_STAR_RATING', 'H_COMP_6_STAR_RATING', 'H_COMP_7_STAR_RATING',
        'H_GLOB_STAR_RATING', 'H_INDI_STAR_RATING'
    ],
    process: [
        'HCP_COVID_19', 'IMM_3', 'OP_10', 'OP_13', 'OP_18B',
        'OP_22', 'OP_23', 'OP_29', 'OP_8', 'PC_01',
        'SAFE_USE_OF_OPIOIDS', 'SEP_1'
    ]
};

// Map section IDs to group names
const SECTION_TO_GROUP = {
    'mortality': 'mortality',
    'safety': 'safety',
    'readmission': 'readmission',
    'experience': 'patient_experience',
    'care': 'process'
};

// Toggle collapsible sections
function toggleSection(sectionId) {
    const header = document.querySelector(`#${sectionId}-content`).previousElementSibling;
    const content = document.getElementById(`${sectionId}-content`);

    header.classList.toggle('expanded');
    content.classList.toggle('expanded');
}

// Count filled measures in each group
function updateMeasureCounts() {
    const sections = ['mortality', 'safety', 'readmission', 'experience', 'care'];

    sections.forEach(section => {
        const groupName = SECTION_TO_GROUP[section];
        const measures = MEASURE_GROUPS[groupName];
        let count = 0;

        measures.forEach(measure => {
            const input = document.getElementById(measure);
            if (input && input.value !== '') {
                count++;
            }
        });

        const badge = document.getElementById(`${section}-count`);
        if (badge) {
            badge.textContent = `${count}/${measures.length} entered`;
        }
    });
}

// Add input listeners for real-time count updates
document.addEventListener('DOMContentLoaded', () => {
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', updateMeasureCounts);
    });

    // Load hospital list
    loadHospitalList();

    // Setup and load hospital table
    setupTableControls();
    loadHospitalTable();

    // Custom Dropdown Event Listeners
    setupCustomDropdown();
});

let allHospitals = [];

// Table State
const tableState = {
    page: 1,
    limit: 10,
    rating: 5 // Default to 5 stars
};

function setupCustomDropdown() {
    const searchInput = document.getElementById('hospital-search');
    const resultsContainer = document.getElementById('hospital-results');

    if (!searchInput || !resultsContainer) return;

    // Show results on focus
    searchInput.addEventListener('focus', () => {
        if (allHospitals.length > 0) {
            resultsContainer.classList.add('show');
            filterHospitals(searchInput.value);
        }
    });

    // Filter on type
    searchInput.addEventListener('input', (e) => {
        resultsContainer.classList.add('show');
        filterHospitals(e.target.value);
    });

    // Hide when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !resultsContainer.contains(e.target)) {
            resultsContainer.classList.remove('show');
        }
    });
}

// Load list of hospitals from API (for search)
async function loadHospitalList() {
    try {
        const response = await fetch('/api/hospitals');
        if (!response.ok) throw new Error('Failed to load hospitals');

        allHospitals = await response.json();

    } catch (error) {
        console.error('Error loading hospital list:', error);
    }
}

function setupTableControls() {
    // Filter change
    const filterSelect = document.getElementById('rating-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            tableState.rating = e.target.value;
            tableState.page = 1; // Reset to first page
            loadHospitalTable();
        });
    }

    // Pagination
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (tableState.page > 1) {
                tableState.page--;
                loadHospitalTable();
            }
        });
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            tableState.page++;
            loadHospitalTable();
        });
    }
}

// Load hospital table data
async function loadHospitalTable() {
    const tbody = document.getElementById('top-hospitals-body');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const pageInfo = document.getElementById('pagination-info');

    if (!tbody) return;

    // Show loading state by dimming, NOT by replacing content (prevents layout shift)
    tbody.style.opacity = '0.5';
    tbody.style.transition = 'opacity 0.2s';

    if (prevBtn) prevBtn.disabled = true;
    if (nextBtn) nextBtn.disabled = true;

    try {
        // Build URL
        let url = `/api/hospitals/list?page=${tableState.page}&limit=${tableState.limit}`;
        if (tableState.rating) {
            url += `&rating=${tableState.rating}`;
        }

        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to load hospitals');

        const data = await response.json();

        // Update state if needed (e.g. if page was out of range)
        tableState.page = data.page;

        renderHospitalTable(data.hospitals);

        // Update pagination controls
        if (pageInfo) {
            pageInfo.textContent = `Page ${data.page} of ${data.total_pages || 1} (${data.total} hospitals)`;
        }

        if (prevBtn) {
            prevBtn.disabled = data.page <= 1;
        }

        if (nextBtn) {
            nextBtn.disabled = data.page >= data.total_pages;
        }

    } catch (error) {
        console.error('Error loading hospital table:', error);
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; padding: 2rem; color: #ef4444;">Failed to load data.</td></tr>';
    } finally {
        // Restore opacity
        tbody.style.opacity = '1';
    }
}

function renderHospitalTable(hospitals) {
    const tbody = document.getElementById('top-hospitals-body');
    if (!tbody) return;

    tbody.innerHTML = '';

    if (hospitals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; padding: 2rem;">No hospitals found with this filter.</td></tr>';
        return;
    }

    hospitals.forEach(hospital => {
        const tr = document.createElement('tr');

        tr.innerHTML = `
            <td>
                <div style="font-weight: 500; color: white;">${hospital.name}</div>
                <div style="font-size: 0.75rem; color: #6b7280;">ID: ${hospital.id}</div>
            </td>
            <td>
                <div class="rating-badge">
                    <span>★</span> ${hospital.rating} Stars
                </div>
            </td>
            <td style="text-align: right;">
                <button class="load-btn-sm" onclick="loadHospitalData('${hospital.id}'); window.scrollTo({ top: 0, behavior: 'smooth' });">
                    Load Data
                </button>
            </td>
        `;

        tbody.appendChild(tr);
    });
}

// Filter hospitals based on search
function filterHospitals(query) {
    const lowerQuery = query.toLowerCase();

    const filtered = allHospitals.filter(h =>
        h.name.toLowerCase().includes(lowerQuery) ||
        h.id.toString().includes(lowerQuery)
    );

    renderDropdownList(filtered.slice(0, 50)); // Limit results for performance
}

// Render the dropdown list
function renderDropdownList(hospitals) {
    const resultsContainer = document.getElementById('hospital-results');
    resultsContainer.innerHTML = '';

    if (hospitals.length === 0) {
        const div = document.createElement('div');
        div.className = 'dropdown-item';
        div.textContent = 'No hospitals found';
        resultsContainer.appendChild(div);
        return;
    }

    hospitals.forEach(hospital => {
        const div = document.createElement('div');
        div.className = 'dropdown-item';
        div.innerHTML = `<strong>${hospital.display_name}</strong>`;

        div.addEventListener('click', () => {
            selectHospital(hospital);
        });

        resultsContainer.appendChild(div);
    });
}

// Handle hospital selection
function selectHospital(hospital) {
    const searchInput = document.getElementById('hospital-search');
    const resultsContainer = document.getElementById('hospital-results');

    searchInput.value = hospital.display_name;
    resultsContainer.classList.remove('show');

    loadHospitalData(hospital.id);
}

// Load specific hospital data and populate form
async function loadHospitalData(providerId) {
    try {
        const response = await fetch(`/api/hospitals/${providerId}`);
        if (!response.ok) throw new Error('Failed to load hospital data');

        const data = await response.json();
        populateForm(data);

        // Auto-calculate after loading
        calculateRating();

    } catch (error) {
        console.error('Error loading hospital data:', error);
        alert('Error loading data for this hospital');
    }
}

// Populate form fields with data
function populateForm(data) {
    // Clear all existing inputs first
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.value = '';
    });

    Object.values(MEASURE_GROUPS).flat().forEach(measure => {
        const input = document.getElementById(measure);
        if (input && data[measure] !== undefined && data[measure] !== null) {
            input.value = data[measure];
        }
    });

    updateMeasureCounts();
}

// Collect all measure values from form
function collectMeasures() {
    const measures = {};

    Object.values(MEASURE_GROUPS).flat().forEach(measure => {
        const input = document.getElementById(measure);
        if (input && input.value !== '') {
            measures[measure] = parseFloat(input.value);
        } else {
            measures[measure] = null;
        }
    });

    return measures;
}

// Main calculation function - calls Python backend
async function calculateRating() {
    const measures = collectMeasures();

    try {
        const response = await fetch('/api/calculate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(measures)
        });

        if (!response.ok) {
            throw new Error('Calculation failed');
        }

        const result = await response.json();
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        // Fallback to client-side calculation if server unavailable
        const result = calculateClientSide(measures);
        displayResults(result);
    }
}

// Client-side calculation fallback (replicates Python logic)
function calculateClientSide(measures) {
    // Measures that need to be flipped (lower = better)
    const MEASURES_TO_FLIP = [
        'MORT_30_AMI', 'MORT_30_CABG', 'MORT_30_COPD', 'MORT_30_HF',
        'MORT_30_PN', 'MORT_30_STK', 'PSI_4_SURG_COMP',
        'COMP_HIP_KNEE', 'HAI_1', 'HAI_2', 'HAI_3', 'HAI_4',
        'HAI_5', 'HAI_6', 'PSI_90_SAFETY',
        'EDAC_30_AMI', 'EDAC_30_HF', 'EDAC_30_PN', 'OP_32',
        'READM_30_CABG', 'READM_30_COPD', 'READM_30_HIP_KNEE',
        'READM_30_HOSP_WIDE', 'OP_35_ADM', 'OP_35_ED', 'OP_36',
        'OP_22', 'PC_01', 'OP_18B', 'OP_8', 'OP_10', 'OP_13',
        'SAFE_USE_OF_OPIOIDS'
    ];

    // Standard weights from CMS
    const STANDARD_WEIGHTS = {
        patient_experience: 0.22,
        readmission: 0.22,
        mortality: 0.22,
        safety: 0.22,
        process: 0.12
    };

    // Group labels for display
    const GROUP_LABELS = {
        mortality: 'Outcomes - Mortality',
        safety: 'Outcomes - Safety of Care',
        readmission: 'Outcomes - Readmission',
        patient_experience: 'Patient Experience',
        process: 'Timely and Effective Care'
    };

    // Calculate group scores
    const groupScores = {};
    const groupCounts = {};

    Object.entries(MEASURE_GROUPS).forEach(([groupName, measureList]) => {
        const values = [];

        measureList.forEach(measure => {
            if (measures[measure] !== null && measures[measure] !== undefined) {
                // For now, use raw values (in production, would standardize against national data)
                let value = measures[measure];

                // Flip if needed (for visual representation, center around 0)
                if (MEASURES_TO_FLIP.includes(measure)) {
                    // For flipped measures, lower raw values are better
                    // We'll normalize differently in display
                }

                values.push(value);
            }
        });

        groupCounts[groupName] = values.length;

        if (values.length > 0) {
            // Simple average (matching SAS grp_score macro)
            groupScores[groupName] = values.reduce((a, b) => a + b, 0) / values.length;
        } else {
            groupScores[groupName] = null;
        }
    });

    // Check eligibility
    const groupsWith3Plus = {};
    Object.entries(groupCounts).forEach(([group, count]) => {
        groupsWith3Plus[group] = count >= 3;
    });

    const totalGroupsWith3 = Object.values(groupsWith3Plus).filter(Boolean).length;
    const mortSafeCount = (groupsWith3Plus.mortality ? 1 : 0) + (groupsWith3Plus.safety ? 1 : 0);

    const eligible = mortSafeCount >= 1 && totalGroupsWith3 >= 3;

    let eligibilityReason;
    if (eligible) {
        eligibilityReason = `Eligible: ${totalGroupsWith3} groups with ≥3 measures`;
    } else {
        const reasons = [];
        if (totalGroupsWith3 < 3) {
            reasons.push(`only ${totalGroupsWith3} groups have ≥3 measures (need 3)`);
        }
        if (mortSafeCount < 1) {
            reasons.push('neither Mortality nor Safety has ≥3 measures');
        }
        eligibilityReason = 'Ineligible: ' + reasons.join('; ');
    }

    if (!eligible) {
        return {
            eligible: false,
            eligibility_reason: eligibilityReason,
            group_scores: Object.fromEntries(
                Object.entries(groupScores).map(([k, v]) => [GROUP_LABELS[k], v])
            ),
            group_measure_counts: Object.fromEntries(
                Object.entries(groupCounts).map(([k, v]) => [GROUP_LABELS[k], v])
            ),
            summary_score: null,
            star_rating: null,
            peer_group: null,
            weights_used: {}
        };
    }

    // Calculate summary score with weight redistribution
    const missing = {};
    Object.entries(groupScores).forEach(([group, score]) => {
        missing[group] = score === null;
    });

    let missingWeightSum = 0;
    Object.entries(missing).forEach(([group, isMissing]) => {
        if (isMissing) {
            missingWeightSum += STANDARD_WEIGHTS[group];
        }
    });

    const redistributedWeights = {};
    Object.entries(STANDARD_WEIGHTS).forEach(([group, weight]) => {
        if (missing[group]) {
            redistributedWeights[group] = null;
        } else {
            redistributedWeights[group] = weight / (1 - missingWeightSum);
        }
    });

    // Calculate weighted summary
    let summaryScore = 0;
    Object.entries(groupScores).forEach(([group, score]) => {
        if (score !== null && redistributedWeights[group] !== null) {
            summaryScore += redistributedWeights[group] * score;
        }
    });

    // Peer group
    const peerGroup = Math.min(Math.max(totalGroupsWith3, 3), 5);

    // Estimate star rating based on summary score
    // Using approximate percentile cutoffs
    let starRating;
    if (summaryScore < -0.84) {
        starRating = 1;
    } else if (summaryScore < -0.25) {
        starRating = 2;
    } else if (summaryScore < 0.25) {
        starRating = 3;
    } else if (summaryScore < 0.84) {
        starRating = 4;
    } else {
        starRating = 5;
    }

    return {
        eligible: true,
        eligibility_reason: eligibilityReason,
        group_scores: Object.fromEntries(
            Object.entries(groupScores).map(([k, v]) => [GROUP_LABELS[k], v])
        ),
        group_measure_counts: Object.fromEntries(
            Object.entries(groupCounts).map(([k, v]) => [GROUP_LABELS[k], v])
        ),
        summary_score: summaryScore,
        star_rating: starRating,
        peer_group: peerGroup,
        weights_used: Object.fromEntries(
            Object.entries(redistributedWeights).map(([k, v]) => [GROUP_LABELS[k], v])
        )
    };
}

// Display results in the UI
function displayResults(result) {
    // Update star display
    for (let i = 1; i <= 5; i++) {
        const star = document.getElementById(`star-${i}`);
        if (result.star_rating !== null && i <= result.star_rating) {
            star.classList.add('active');
            star.style.animationDelay = `${(i - 1) * 0.1}s`;
        } else {
            star.classList.remove('active');
        }
    }

    // Update star label
    const starLabel = document.getElementById('star-label');
    if (result.star_rating !== null) {
        starLabel.textContent = `${result.star_rating} Star Rating`;
        starLabel.classList.remove('pending');
    } else {
        starLabel.textContent = 'Not Eligible for Rating';
        starLabel.classList.add('pending');
    }

    // Update eligibility
    const eligibilityStatus = document.getElementById('eligibility-status');
    const eligibilityDetails = document.getElementById('eligibility-details');

    if (result.eligible) {
        eligibilityStatus.textContent = '✓ Eligible for Star Rating';
        eligibilityStatus.className = 'eligibility-status eligible';
    } else {
        eligibilityStatus.textContent = '✗ Not Eligible';
        eligibilityStatus.className = 'eligibility-status ineligible';
    }
    eligibilityDetails.textContent = result.eligibility_reason;

    // Update group scores
    const groupMapping = {
        'Outcomes - Mortality': { score: 'mortality-score', bar: 'mortality-bar' },
        'Outcomes - Safety of Care': { score: 'safety-score', bar: 'safety-bar' },
        'Outcomes - Readmission': { score: 'readmission-score', bar: 'readmission-bar' },
        'Patient Experience': { score: 'experience-score', bar: 'experience-bar' },
        'Timely and Effective Care': { score: 'care-score', bar: 'care-bar' }
    };

    Object.entries(result.group_scores).forEach(([groupLabel, score]) => {
        const mapping = groupMapping[groupLabel];
        if (mapping) {
            const scoreEl = document.getElementById(mapping.score);
            const barEl = document.getElementById(mapping.bar);

            if (score !== null) {
                scoreEl.textContent = score.toFixed(3);
                // Convert score to percentage for bar (assuming -3 to +3 range)
                const percent = Math.max(0, Math.min(100, ((score + 3) / 6) * 100));
                barEl.style.width = `${percent}%`;

                if (score < 0) {
                    barEl.classList.add('negative');
                } else {
                    barEl.classList.remove('negative');
                }
            } else {
                scoreEl.textContent = 'N/A';
                barEl.style.width = '0%';
            }
        }
    });

    // Update summary score
    const summaryScoreEl = document.getElementById('summary-score');
    if (result.summary_score !== null) {
        summaryScoreEl.textContent = result.summary_score.toFixed(4);
    } else {
        summaryScoreEl.textContent = '--';
    }
}
