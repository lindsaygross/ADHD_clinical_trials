"""
Fetch ADHD clinical trial data from ClinicalTrials.gov API.

This module retrieves interventional Phase 1, 2, and 3 trials related to ADHD
and saves the raw data for further processing.
"""

import json
import os
import time
from typing import Dict, List, Any

import pandas as pd
import requests

# ClinicalTrials.gov API v2 endpoint
API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_adhd_trials(
        max_results: int = 1000,
        page_size: int = 100,
        delay_seconds: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fetch ADHD interventional trials from ClinicalTrials.gov API v2.

    Parameters
    ----------
    max_results : int
        Maximum number of results to retrieve
    page_size : int
        Number of results per page (API limits to ~100)
    delay_seconds : float
        Delay between API requests to be respectful

    Returns
    -------
    List[Dict[str, Any]]
        List of trial records
    """
    all_trials = []
    next_page_token = None

    # Build query parameters
    # Using API v2 query syntax - simpler query format
    query = {
        "query.cond": "ADHD",
        "query.intr": "Interventional",
        "pageSize": page_size,
        "format": "json",
    }

    print(f"Fetching ADHD interventional trials from ClinicalTrials.gov...")

    page = 1
    while len(all_trials) < max_results:
        # Add page token if we have one
        params = query.copy()
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract studies from response
            studies = data.get("studies", [])

            if not studies:
                print(f"No more studies found. Stopping at page {page}.")
                break

            all_trials.extend(studies)
            print(f"Page {page}: Retrieved {len(studies)} trials (total: {len(all_trials)})")

            # Check if there's a next page
            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                print("No more pages available.")
                break

            page += 1

            # Respectful delay between requests
            time.sleep(delay_seconds)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    print(f"\nTotal trials retrieved: {len(all_trials)}")
    return all_trials


def extract_fields_from_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant fields from a trial record.

    The API v2 structure nests data differently than v1.

    Parameters
    ----------
    trial : Dict[str, Any]
        Raw trial record from API

    Returns
    -------
    Dict[str, Any]
        Flattened trial data
    """
    protocol = trial.get("protocolSection", {})

    # Identification
    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId", "")
    brief_title = id_module.get("briefTitle", "")
    official_title = id_module.get("officialTitle", "")

    # Status
    status_module = protocol.get("statusModule", {})
    overall_status = status_module.get("overallStatus", "")
    start_date = status_module.get("startDateStruct", {}).get("date", "")
    completion_date = status_module.get("completionDateStruct", {}).get("date", "")

    # Sponsor
    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsor_module.get("leadSponsor", {})
    lead_sponsor_name = lead_sponsor.get("name", "")
    lead_sponsor_class = lead_sponsor.get("class", "")

    # Design
    design_module = protocol.get("designModule", {})
    study_type = design_module.get("studyType", "")
    phases = design_module.get("phases", [])
    phase = ", ".join(phases) if phases else ""
    enrollment_count = design_module.get("enrollmentInfo", {}).get("count")

    design_info = design_module.get("designInfo", {})
    allocation = design_info.get("allocation", "")
    intervention_model = design_info.get("interventionModel", "")
    primary_purpose = design_info.get("primaryPurpose", "")
    masking = design_info.get("maskingInfo", {}).get("masking", "")
    masking_desc = design_info.get("maskingInfo", {}).get("maskingDescription", "")

    # Arms
    arms_module = protocol.get("armsInterventionsModule", {})
    arms = arms_module.get("armGroups", [])
    number_of_arms = len(arms) if arms else None

    # Interventions
    interventions = arms_module.get("interventions", [])
    intervention_types = [iv.get("type", "") for iv in interventions]
    intervention_names = [iv.get("name", "") for iv in interventions]

    # Outcomes
    outcomes_module = protocol.get("outcomesModule", {})
    primary_outcomes = outcomes_module.get("primaryOutcomes", [])
    primary_outcome_measures = [po.get("measure", "") for po in primary_outcomes]
    primary_outcome_descriptions = [po.get("description", "") for po in primary_outcomes]

    # Conditions
    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])

    # Eligibility
    eligibility_module = protocol.get("eligibilityModule", {})
    min_age = eligibility_module.get("minimumAge", "")
    max_age = eligibility_module.get("maximumAge", "")
    gender = eligibility_module.get("sex", "")
    healthy_volunteers = eligibility_module.get("healthyVolunteers", "")

    # Locations
    locations_module = protocol.get("contactsLocationsModule", {})
    locations = locations_module.get("locations", [])
    countries = list(set([loc.get("country", "") for loc in locations if loc.get("country")]))

    return {
        "NCTId": nct_id,
        "BriefTitle": brief_title,
        "OfficialTitle": official_title,
        "OverallStatus": overall_status,
        "StudyType": study_type,
        "Phase": phase,
        "EnrollmentCount": enrollment_count,
        "StartDate": start_date,
        "CompletionDate": completion_date,
        "LeadSponsorName": lead_sponsor_name,
        "LeadSponsorClass": lead_sponsor_class,
        "DesignAllocation": allocation,
        "DesignInterventionModel": intervention_model,
        "DesignPrimaryPurpose": primary_purpose,
        "DesignMasking": masking,
        "DesignMaskingDescription": masking_desc,
        "NumberOfArms": number_of_arms,
        "InterventionType": "; ".join(intervention_types),
        "InterventionName": "; ".join(intervention_names),
        "PrimaryOutcomeMeasure": "; ".join(primary_outcome_measures),
        "PrimaryOutcomeDescription": "; ".join(primary_outcome_descriptions),
        "Condition": "; ".join(conditions),
        "LocationCountry": "; ".join(countries),
        "MinimumAge": min_age,
        "MaximumAge": max_age,
        "Gender": gender,
        "HealthyVolunteers": healthy_volunteers,
    }


def filter_target_phases(trials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter trials to include Phase 1, Phase 2, and Phase 3.

    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame of all trials

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    # Updated Regex to include Phase 1
    # This matches: "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3"
    phase_mask = trials_df["Phase"].fillna("").str.contains(
        "PHASE1|PHASE2|PHASE3|Phase 1|Phase 2|Phase 3",
        case=False,
        regex=True
    )

    filtered_df = trials_df[phase_mask].copy()
    print(f"\nFiltered to Phase 1/2/3 trials: {len(filtered_df)} trials")

    return filtered_df


def save_data(trials: List[Dict[str, Any]], output_dir: str = "data/raw"):
    """
    Save trial data in both JSON and CSV formats.

    Parameters
    ----------
    trials : List[Dict[str, Any]]
        List of trial records
    output_dir : str
        Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw JSON
    json_path = os.path.join(output_dir, "adhd_trials_raw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw JSON to: {json_path}")

    # Extract and flatten fields
    extracted_trials = [extract_fields_from_trial(trial) for trial in trials]
    df = pd.DataFrame(extracted_trials)

    # Filter to Phases 1, 2, and 3
    df_filtered = filter_target_phases(df)

    # Save full CSV
    # Renamed output file to reflect new phase inclusion
    csv_path = os.path.join(output_dir, "adhd_trials_raw.csv")
    df_filtered.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved filtered CSV to: {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total trials (Phase 1/2/3): {len(df_filtered)}")
    print(f"\nStatus distribution:")
    print(df_filtered["OverallStatus"].value_counts())
    print(f"\nPhase distribution:")
    print(df_filtered["Phase"].value_counts())
    print(f"\nSponsor class distribution:")
    print(df_filtered["LeadSponsorClass"].value_counts())

    return df_filtered


def main():
    """Main execution function."""
    print("=" * 60)
    print("ADHD Clinical Trials Data Fetcher (Expanded)")
    print("=" * 60)

    # Fetch trials
    trials = fetch_adhd_trials(max_results=2000, page_size=100)

    if not trials:
        print("No trials retrieved. Exiting.")
        return

    # Save data
    df = save_data(trials)

    print("\n" + "=" * 60)
    print("Data fetching complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
