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

// Tooltip descriptions for each measure (from CMS specification)
const MEASURE_TOOLTIPS = {
    // Mortality Group
    'MORT_30_AMI': {
        description: 'AMI 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-AMI'
    },
    'MORT_30_CABG': {
        description: 'CABG 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-CABG'
    },
    'MORT_30_COPD': {
        description: 'COPD 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-COPD'
    },
    'MORT_30_HF': {
        description: 'HF 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-HF'
    },
    'MORT_30_PN': {
        description: 'Pneumonia 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-PN'
    },
    'MORT_30_STK': {
        description: 'Stroke 30-Day Mortality Risk-Standardized Mortality Rate',
        measureId: 'MORT-30-STK'
    },
    'PSI_4_SURG_COMP': {
        description: 'PSI-4 smoothed measure rate per 1,000 eligible discharges (Death Rate Among Surgical Patients)',
        measureId: 'PSI-4'
    },

    // Safety Group
    'COMP_HIP_KNEE': {
        description: 'THA/TKA Complication Risk-Standardized Complication Rate',
        measureId: 'COMP-HIP-KNEE'
    },
    'HAI_1': {
        description: 'CLABSI (Central Line-Associated Bloodstream Infection) in ICU + Select Wards - Standardized Infection Ratio',
        measureId: 'HAI-1'
    },
    'HAI_2': {
        description: 'CAUTI (Catheter-Associated Urinary Tract Infection) in ICU + Select Wards - Standardized Infection Ratio',
        measureId: 'HAI-2'
    },
    'HAI_3': {
        description: 'SSI - Colon Surgery Standardized Infection Ratio',
        measureId: 'HAI-3'
    },
    'HAI_4': {
        description: 'SSI - Abdominal Hysterectomy Standardized Infection Ratio',
        measureId: 'HAI-4'
    },
    'HAI_5': {
        description: 'MRSA Bacteremia Standardized Infection Ratio',
        measureId: 'HAI-5'
    },
    'HAI_6': {
        description: 'C.Diff (Clostridioides difficile) Standardized Infection Ratio',
        measureId: 'HAI-6'
    },
    'PSI_90_SAFETY': {
        description: 'PSI-90 composite value (Patient Safety Indicator composite)',
        measureId: 'PSI-90'
    },

    // Readmission Group
    'EDAC_30_AMI': {
        description: 'AMI 30-Excess Days in Acute Care Rate',
        measureId: 'EDAC-30-AMI'
    },
    'EDAC_30_HF': {
        description: 'HF 30-Excess Days in Acute Care Rate',
        measureId: 'EDAC-30-HF'
    },
    'EDAC_30_PN': {
        description: 'Excess Days in Acute Care (EDAC) after hospitalization for Pneumonia (PN)',
        measureId: 'EDAC-30-PN'
    },
    'OP_32': {
        description: 'OP-32 measure rate (Facility 7-Day Risk-Standardized Hospital Visit Rate after Outpatient Colonoscopy)',
        measureId: 'OP-32'
    },
    'READM_30_CABG': {
        description: 'CABG 30-Day Readmission Risk-Standardized Readmission Rate',
        measureId: 'READM-30-CABG'
    },
    'READM_30_COPD': {
        description: 'COPD 30-Day Readmission Risk-Standardized Readmission Rate',
        measureId: 'READM-30-COPD'
    },
    'READM_30_HIP_KNEE': {
        description: 'THA/TKA 30-Day Readmission Risk-Standardized Readmission Rate',
        measureId: 'READM-30-HIP-KNEE'
    },
    'READM_30_HOSP_WIDE': {
        description: 'Hospital Wide Readmission Risk-Standardized Readmission Rate',
        measureId: 'READM-30-HOSPWIDE'
    },
    'OP_35_ADM': {
        description: 'Admissions for Patients Receiving Outpatient Chemotherapy',
        measureId: 'OP-35 ADM'
    },
    'OP_35_ED': {
        description: 'ED Visits for Patients Receiving Outpatient Chemotherapy',
        measureId: 'OP-35 ED'
    },
    'OP_36': {
        description: 'Hospital Visits after Hospital Outpatient Surgery',
        measureId: 'OP-36'
    },

    // Patient Experience Group
    'H_COMP_1_STAR_RATING': {
        description: 'HCAHPS Composite 1-star rating (Q1 to Q3) - Communication with Nurses',
        measureId: 'Composite 1 Q1 to Q3'
    },
    'H_COMP_2_STAR_RATING': {
        description: 'HCAHPS Composite 2-star rating (Q5 to Q7) - Communication with Doctors',
        measureId: 'Composite 2 Q5 to Q7'
    },
    'H_COMP_3_STAR_RATING': {
        description: 'HCAHPS Composite 3-star rating (Q4 & Q11) - Staff Responsiveness',
        measureId: 'Composite 3 Q4 & Q11'
    },
    'H_COMP_5_STAR_RATING': {
        description: 'HCAHPS Composite 5-star rating (Q16 to Q17) - Communication About Medicines',
        measureId: 'Composite 5 Q16 & Q17'
    },
    'H_COMP_6_STAR_RATING': {
        description: 'HCAHPS Composite 6-star rating (Q19 to Q20) - Discharge Information',
        measureId: 'Composite 6 Q19 & Q20'
    },
    'H_COMP_7_STAR_RATING': {
        description: 'HCAHPS Composite 7-star rating (Q23 to Q25) - Care Transition',
        measureId: 'Composite 7 Q23 to Q25'
    },
    'H_GLOB_STAR_RATING': {
        description: '(H-HSP-RATING Overall Rating of Hospital Q21 + H-RECMND Willingness to Recommend Hospital Q22) / 2',
        measureId: 'Composite Q21 & Q22'
    },
    'H_INDI_STAR_RATING': {
        description: '(H-CLEAN-HSP Cleanliness of Hospital Environment Q8 + H-QUIET-HSP Quietness of Hospital Environment Q9) / 2',
        measureId: 'Composite Q8 & Q9'
    },

    // Timely & Effective Care Group
    'HCP_COVID_19': {
        description: 'COVID-19 Vaccination Coverage Among Healthcare Personnel',
        measureId: 'HCP_COVID_19'
    },
    'IMM_3': {
        description: 'IMM-3 measure rate (Influenza vaccination adherence percentage). IMM-3 and OP-27 are the same data for a hospital.',
        measureId: 'IMM-3'
    },
    'OP_10': {
        description: 'OP-10 measure rate (Abdomen CT - Use of Contrast Material)',
        measureId: 'OP-10'
    },
    'OP_13': {
        description: 'OP-13 measure rate (Cardiac Imaging for Preoperative Risk Assessment for Non-Cardiac Low-Risk Surgery)',
        measureId: 'OP-13'
    },
    'OP_18B': {
        description: 'OP-18b: Median time from ED arrival to ED departure for discharged ED patients',
        measureId: 'OP-18b'
    },
    'OP_22': {
        description: 'OP-22 measure rate (Left Without Being Seen)',
        measureId: 'OP-22'
    },
    'OP_23': {
        description: 'OP-23 measure rate (ED Head CT or MRI Scan Results for Acute Ischemic Stroke or Hemorrhagic Stroke)',
        measureId: 'OP-23'
    },
    'OP_29': {
        description: 'OP-29 rate (Appropriate Follow-Up Interval for Normal Colonoscopy in Average Risk Patients)',
        measureId: 'OP-29'
    },
    'OP_8': {
        description: 'OP-8 measure rate (MRI Lumbar Spine for Low Back Pain)',
        measureId: 'OP-8'
    },
    'PC_01': {
        description: 'PC-01 measure rate (Elective Delivery Prior to 39 Completed Weeks Gestation)',
        measureId: 'PC-01'
    },
    'SAFE_USE_OF_OPIOIDS': {
        description: 'Percentage of patients who were prescribed 2 or more opioids or an opioid and benzodiazepine concurrently at discharge',
        measureId: 'Safe Use of Opioids'
    },
    'SEP_1': {
        description: 'SEP-1 measure rate (Severe Sepsis and Septic Shock: Management Bundle)',
        measureId: 'SEP-1'
    }
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
document.addEventListener('DOMContentLoaded', async () => {
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', updateMeasureCounts);
    });

    // Apply tooltips to static measure labels
    applyMeasureTooltips();

    // Load hospital list FIRST to ensure we have names for lookup
    await loadHospitalList();

    // Autofill University of Kentucky (180067)
    loadHospitalData('180067');

    // Setup and load hospital table
    setupTableControls();
    loadHospitalTable();

    // Custom Dropdown Event Listeners
    setupCustomDropdown();

    // Load optimization config (measures)
    loadOptimizationConfig();
});

// Apply tooltips to static measure labels in the form
function applyMeasureTooltips() {
    // Find all input groups and add tooltips to their labels
    const allMeasures = Object.values(MEASURE_GROUPS).flat();

    allMeasures.forEach(measure => {
        const input = document.getElementById(measure);
        if (!input) return;

        const inputGroup = input.closest('.input-group');
        if (!inputGroup) return;

        const label = inputGroup.querySelector('label');
        if (!label) return;

        const tooltip = MEASURE_TOOLTIPS[measure];
        if (!tooltip) return;

        // Create tooltip wrapper
        const wrapper = document.createElement('span');
        wrapper.className = 'tooltip-container';
        wrapper.innerHTML = `
            <span class="tooltip-icon">?</span>
            <span class="tooltip-text">
                ${tooltip.description}
                <br><span class="tooltip-measure-id">${tooltip.measureId}</span>
            </span>
        `;

        // Append to label
        label.appendChild(wrapper);
    });
}

let actionableMeasures = []; // specific measures from backend
let measureMeta = {}; // Friendly names helper

async function loadOptimizationConfig() {
    try {
        const response = await fetch('/api/measures');
        const data = await response.json();

        actionableMeasures = data.actionable || [];
        // Flatten groups for easy access to measure definitions if needed, 
        // but here we just need to list them.

        // We need all measures list.
        const allMeasures = Object.values(data.groups).flat();

        const actionableBody = document.getElementById('opt-actionable-tbody');
        const otherBody = document.getElementById('opt-other-tbody');

        if (actionableBody) actionableBody.innerHTML = '';
        if (otherBody) otherBody.innerHTML = '';

        allMeasures.forEach(measure => {
            const isActionable = actionableMeasures.includes(measure);

            const tr = document.createElement('tr');
            // Checkbox
            const tdCheck = document.createElement('td');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'opt-measure';
            checkbox.value = measure;
            checkbox.checked = isActionable; // Default check if actionable
            checkbox.id = `opt-check-${measure}`;
            tdCheck.appendChild(checkbox);
            tr.appendChild(tdCheck);

            // Measure Name with Tooltip
            const tdName = document.createElement('td');
            const tooltip = MEASURE_TOOLTIPS[measure];
            if (tooltip) {
                tdName.innerHTML = `
                    <div class="opt-measure-label">
                        <label for="opt-check-${measure}" style="cursor:pointer; color: #cbd5e1; font-weight:500;">${measure}</label>
                        <span class="tooltip-container">
                            <span class="tooltip-icon">?</span>
                            <span class="tooltip-text">
                                ${tooltip.description}
                                <br><span class="tooltip-measure-id">${tooltip.measureId}</span>
                            </span>
                        </span>
                    </div>`;
            } else {
                tdName.innerHTML = `<label for="opt-check-${measure}" style="cursor:pointer; color: #cbd5e1; font-weight:500;">${measure}</label>`;
            }
            tr.appendChild(tdName);

            // Cost Input
            const tdCost = document.createElement('td');
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `opt-cost-${measure}`;
            input.value = 5000; // Default cost
            input.step = 1000;
            input.style = "width: 100px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 4px; border-radius: 4px; text-align: right;";
            tdCost.appendChild(input);
            tr.appendChild(tdCost);

            if (isActionable && actionableBody) {
                actionableBody.appendChild(tr);
            } else if (otherBody) {
                otherBody.appendChild(tr);
            }
        });

    } catch (e) {
        console.error("Error loading optimization config:", e);
    }
}

function toggleTargetScoreInput() {
    const radios = document.getElementsByName('target_mode');
    let mode = 'next_star';
    for (const r of radios) if (r.checked) mode = r.value;

    const container = document.getElementById('target-score-container');
    if (container) {
        container.style.display = (mode === 'specific_score') ? 'block' : 'none';
        if (mode === 'specific_score') {
            document.getElementById('target_score').focus();
        }
    }
}

function toggleTable(tbodyId) {
    // Specifically for the "Other" container
    if (tbodyId === 'opt-other-tbody') {
        const container = document.getElementById('opt-other-container');
        if (container) {
            const isHidden = container.style.display === 'none';
            container.style.display = isHidden ? 'block' : 'none';
        }
    }
}

function toggleAll(type, state) {
    if (type === 'actionable') {
        const tbody = document.getElementById('opt-actionable-tbody');
        if (tbody) {
            tbody.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = state);
        }
    }
}

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

        // Update Heading Name
        const heading = document.getElementById('hospital-name-heading');
        if (heading) {
            let name = data['Facility Name'] || data['facility_name'] || data['name'];

            // Fallback: Try to find name in the loaded list if not in data
            if (!name && allHospitals.length > 0) {
                const found = allHospitals.find(h => h.id == providerId);
                if (found) name = found.display_name || found.name;
            }

            // Final fallback
            name = name || "Hospital";

            heading.textContent = `Star Rating for ${name}`;
        }

        populateForm(data);

        // Enable Optimize Button
        const btnOptimize = document.getElementById('btn-optimize');
        if (btnOptimize) {
            btnOptimize.disabled = false;
            btnOptimize.style.opacity = '1';
            btnOptimize.style.cursor = 'pointer';
        }

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
        if (input.id !== 'budget-input' && input.id !== 'cost-input' && input.id !== 'target_score') {
            input.value = '';
        }
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
        'Outcomes - Mortality': { score: 'mortality-score', bar: 'mortality-bar', cls: 'mortality' },
        'Outcomes - Safety of Care': { score: 'safety-score', bar: 'safety-bar', cls: 'safety' },
        'Outcomes - Readmission': { score: 'readmission-score', bar: 'readmission-bar', cls: 'readmission' },
        'Patient Experience': { score: 'experience-score', bar: 'experience-bar', cls: 'experience' },
        'Timely and Effective Care': { score: 'care-score', bar: 'care-bar', cls: 'care' }
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

                // Always apply group-specific color
                barEl.className = `score-fill ${mapping.cls || ''}`;
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

    // Enable Optimize Button if we have a valid result
    const btnOptimize = document.getElementById('btn-optimize');
    if (btnOptimize) {
        // If we have a star rating or at least a summary score, enable optimization
        if (result.star_rating !== null || result.summary_score !== null) {
            btnOptimize.disabled = false;
            btnOptimize.style.opacity = '1';
            btnOptimize.style.cursor = 'pointer';
        }
    }
}

// ==========================================
// Optimization Feature
// ==========================================

let currentRecommendations = [];

async function optimizeRating() {
    // Collect settings
    const budgetInput = document.getElementById('budget-input');
    if (!budgetInput || !budgetInput.value) {
        alert("Please enter a Budget Limit.");
        return;
    }
    const budget = parseFloat(budgetInput.value);

    // Target Mode
    const radios = document.getElementsByName('target_mode');
    let targetMode = 'next_star';
    for (const r of radios) if (r.checked) targetMode = r.value;

    let targetScore = null;
    if (targetMode === 'specific_score') {
        const tsInput = document.getElementById('target_score');
        if (tsInput && tsInput.value) targetScore = parseFloat(tsInput.value);
    }

    // Measure Costs & Selection
    // Iterate all checked boxes
    const measureCosts = {};
    const checkedBoxes = document.querySelectorAll('input[name="opt-measure"]:checked');

    if (checkedBoxes.length === 0) {
        alert("Please select at least one measure to optimize.");
        return;
    }

    checkedBoxes.forEach(cb => {
        const m = cb.value;
        const costInput = document.getElementById(`opt-cost-${m}`);
        const cost = costInput ? parseFloat(costInput.value) : 5000.0;
        measureCosts[m] = cost;
    });

    // Also collect current values
    const measures = collectMeasures();

    // Show Modal & Loading
    const modal = document.getElementById('optimization-modal');
    const loading = document.getElementById('optimization-loading');
    const results = document.getElementById('optimization-results');

    if (modal) {
        modal.style.display = 'flex';
        if (loading) loading.style.display = 'block';
        if (results) results.style.display = 'none';
    }

    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...measures,
                budget: budget,
                measure_costs: measureCosts,
                target_mode: targetMode,
                target_score: targetScore
            })
        });

        if (!response.ok) throw new Error("Optimization failed");

        const data = await response.json();

        // Hide loading, show results
        if (loading) loading.style.display = 'none';
        if (results) results.style.display = 'block';

        // Populate UI
        const elCurrentStars = document.getElementById('opt-current-stars');
        if (elCurrentStars) elCurrentStars.textContent = (data.original_rating || '--') + ' Stars';

        const elCurrentScore = document.getElementById('opt-current-score');
        if (elCurrentScore) elCurrentScore.textContent = 'Score: ' + (data.original_summary_score ? data.original_summary_score.toFixed(3) : '--');

        const elProjStars = document.getElementById('opt-projected-stars');
        if (elProjStars) elProjStars.textContent = (data.optimized_rating || '--') + ' Stars';

        const elProjScore = document.getElementById('opt-projected-score');
        if (elProjScore) elProjScore.textContent = 'Score: ' + (data.optimized_summary_score ? data.optimized_summary_score.toFixed(3) : '--');

        const elTotalCost = document.getElementById('opt-total-cost');
        if (elTotalCost) elTotalCost.textContent = '$' + data.total_cost.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 });

        const tbody = document.getElementById('recommendations-body');
        if (tbody) {
            tbody.innerHTML = '';

            currentRecommendations = data.recommendations; // Store for apply

            if (!data.recommendations || data.recommendations.length === 0) {
                tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 1rem; color: #9ca3af;">No changes recommended within budget.</td></tr>';
            } else {
                data.recommendations.forEach(rec => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="color: #fff;">
                            <div style="font-weight: 500;">${rec.measure}</div>
                            <div style="font-size: 0.75rem; color: #6b7280;">Current: ${rec.current_value.toFixed(2)} → Target: ${rec.target_value.toFixed(2)}</div>
                        </td>
                        <td style="color: #cbd5e1;">${rec.description}</td>
                        <td style="text-align: right; color: #ef4444; font-weight: 500;">$${rec.cost.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        }

    } catch (e) {
        console.error(e);
        alert("Optimization failed: " + e.message);
        closeModal();
    }
}

function closeModal() {
    const modal = document.getElementById('optimization-modal');
    if (modal) modal.style.display = 'none';
}

function applyOptimization() {
    if (!currentRecommendations || currentRecommendations.length === 0) {
        closeModal();
        return;
    }

    let updateCount = 0;
    currentRecommendations.forEach(rec => {
        const input = document.getElementById(rec.measure);
        if (input) {
            input.value = rec.target_value; // Updates value
            updateCount++;
        }
    });

    // Update counts manually since programmatic change doesn't always trigger 'input' event
    updateMeasureCounts();

    alert(`Updated ${updateCount} measures with optimized values.`);
    closeModal();
    calculateRating(); // Re-run main calculation
}

// Close modal if clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('optimization-modal');
    if (event.target == modal) {
        closeModal();
    }
}

