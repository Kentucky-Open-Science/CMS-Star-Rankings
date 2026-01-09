# CMS Star Rankings

A web-based tool for calculating and exploring CMS Hospital Star Ratings, replicating the official SAS methodology (July 2025). Data here: https://qualitynet.cms.gov/inpatient/public-reporting/overall-ratings/sas

## Overview

This project converts the official CMS SAS code for Hospital Star Ratings into a modern Python implementation wrapped in a Flask web application. It allows users to:
- Browse and search for hospitals.
- View detailed quality measures for specific hospitals.
- Calculate Star Ratings dynamically based on input measures.
- Visualize the breakdown of scores across different measure groups (Mortality, Safety of Care, Readmission, Patient Experience, and Timely & Effective Care).

The calculation logic is a direct port of the Yale CORE SAS Package (July 2025), ensuring accuracy with official CMS methods.

## Prerequisites

- Python 3.8 or higher
- `pip` (Python package installer)

## Installation

1. Clone or download this repository.
2. Navigate to the project directory:
   ```bash
   cd "CMS Star Rankings"
   ```
3. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
4. Install the required Python packages:
   ```bash
   pip install -r star_ratings_app/requirements.txt
   ```

## Usage

1. Navigate to the application directory:
   ```bash
   cd star_ratings_app
   ```
2. Run the application:
   ```bash
   python calculations.py
   ```
3. Open your web browser and go to:
   [http://localhost:5555](http://localhost:5555)

## Project Structure

- `star_ratings_app/`: Contains the Flask application source code.
  - `calculations.py`: Main application logic, including the Star Rating algorithm and Flask routes.
  - `index.html`: Frontend user interface.
  - `script.js`: Client-side logic for interactivity and API communication.
  - `style.css`: Application styling.
  - `requirements.txt`: Python dependencies.
- `alldata_2025jul.csv`: Source data file containing hospital measures (located in the root directory).
- `Hospital_General_Information.csv`: Source data file containing hospital names and general info (located in the root directory).

## Methodology

The application implements the following steps from the CMS methodology:
1. **Standardization**: Converting raw measure scores to Z-scores using national means and standard deviations.
2. **Grouping**: Categorizing 46 measures into 5 groups (Mortality, Safety, Readmission, Patient Experience, Process).
3. **Weighting**: Applying standard weights to group scores.
4. **Clustering**: Using K-means clustering to determine star rating cutoffs (1-5 stars) based on summary scores within peer groups.

## License

See the [LICENSE](LICENSE) file for details.
