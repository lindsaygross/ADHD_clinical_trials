"""
Prepare and label ADHD clinical trial data for machine learning.

This module:
1. Loads raw trial data (Phase 1, 2, and 3)
2. Creates binary labels (success vs failure)
3. Engineers features using only pre-trial information
4. Handles missing data
5. Saves processed dataset
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


def load_raw_data(file_path: str = "data/raw/adhd_trials_raw.csv") -> pd.DataFrame:
    """
    Load raw trial data from CSV.

    Parameters
    ----------
    file_path : str
        Path to raw data CSV

    Returns
    -------
    pd.DataFrame
        Raw trial data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} trials from {file_path}")
    return df


def create_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary success/failure labels.

    Success (1): COMPLETED
    Failure (0): TERMINATED, WITHDRAWN, SUSPENDED

    Exclude trials with other statuses (ACTIVE, RECRUITING, etc.)

    Parameters
    ----------
    df : pd.DataFrame
        Raw trial data

    Returns
    -------
    pd.DataFrame
        Data with success labels
    """
    df = df.copy()

    # Define success and failure statuses
    success_statuses = ["COMPLETED"]
    failure_statuses = ["TERMINATED", "WITHDRAWN", "SUSPENDED"]

    # Create label column
    df["Label"] = None
    df.loc[df["OverallStatus"].isin(success_statuses), "Label"] = 1
    df.loc[df["OverallStatus"].isin(failure_statuses), "Label"] = 0

    # Filter to only labeled trials
    df_labeled = df[df["Label"].notna()].copy()
    df_labeled["Label"] = df_labeled["Label"].astype(int)

    print(f"\n{'='*60}")
    print("LABEL DISTRIBUTION")
    print(f"{'='*60}")
    print(f"Total labeled trials: {len(df_labeled)}")
    print(f"Successful (Completed): {(df_labeled['Label'] == 1).sum()} "
          f"({(df_labeled['Label'] == 1).sum() / len(df_labeled) * 100:.1f}%)")
    print(f"Failed (Terminated/Withdrawn/Suspended): {(df_labeled['Label'] == 0).sum()} "
          f"({(df_labeled['Label'] == 0).sum() / len(df_labeled) * 100:.1f}%)")

    # Show excluded statuses
    excluded = df[df["Label"].isna()]
    if len(excluded) > 0:
        print(f"\nExcluded {len(excluded)} trials with statuses:")
        print(excluded["OverallStatus"].value_counts())

    return df_labeled


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features using only pre-trial information.

    No data leakage - only use information available before/at trial start.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled trial data

    Returns
    -------
    pd.DataFrame
        Data with engineered features
    """
    df = df.copy()

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING")
    print(f"{'='*60}")

    # --- Enrollment features ---
    df["EnrollmentCount"] = pd.to_numeric(df["EnrollmentCount"], errors="coerce")
    df["LogEnrollment"] = np.log1p(df["EnrollmentCount"])
    df["SmallTrial"] = (df["EnrollmentCount"] < 50).astype(int)
    df["LargeTrial"] = (df["EnrollmentCount"] >= 200).astype(int)

    # --- Phase features ---
    df["IsPhase1"] = df["Phase"].fillna("").str.contains("PHASE1|Phase 1", case=False, regex=True).astype(int)
    df["IsPhase2"] = df["Phase"].fillna("").str.contains("PHASE2|Phase 2", case=False, regex=True).astype(int)
    df["IsPhase3"] = df["Phase"].fillna("").str.contains("PHASE3|Phase 3", case=False, regex=True).astype(int)
    df["IsPhase2And3"] = (df["IsPhase2"] & df["IsPhase3"]).astype(int)

    # --- Design features ---
    # Allocation
    df["IsRandomized"] = df["DesignAllocation"].fillna("").str.contains(
        "RANDOMIZED|Randomized", case=False, regex=True
    ).astype(int)

    # Masking/Blinding
    df["MaskingType"] = df["DesignMasking"].fillna("NONE")
    df["IsDoubleBlind"] = df["MaskingType"].str.contains("DOUBLE", case=False, na=False).astype(int)
    df["IsBlinded"] = (~df["MaskingType"].str.contains("NONE|None", case=False, na=True)).astype(int)

    # Number of arms
    df["NumberOfArms"] = pd.to_numeric(df["NumberOfArms"], errors="coerce")
    df["HasMultipleArms"] = (df["NumberOfArms"] > 1).astype(int)
    df["NumArms_2"] = (df["NumberOfArms"] == 2).astype(int)
    df["NumArms_3Plus"] = (df["NumberOfArms"] >= 3).astype(int)

    # Intervention model
    df["IsParallelAssignment"] = df["DesignInterventionModel"].fillna("").str.contains(
        "PARALLEL|Parallel", case=False, regex=True
    ).astype(int)
    df["IsCrossover"] = df["DesignInterventionModel"].fillna("").str.contains(
        "CROSSOVER|Crossover", case=False, regex=True
    ).astype(int)

    # --- Intervention features ---
    df["InterventionType"] = df["InterventionType"].fillna("")
    df["IsDrugIntervention"] = df["InterventionType"].str.contains("DRUG|Drug", case=False, na=False).astype(int)
    df["IsBehavioralIntervention"] = df["InterventionType"].str.contains(
        "BEHAVIORAL|Behavioral", case=False, na=False
    ).astype(int)
    df["IsDeviceIntervention"] = df["InterventionType"].str.contains(
        "DEVICE|Device", case=False, na=False
    ).astype(int)

    # Count number of interventions
    df["NumInterventions"] = df["InterventionType"].apply(
        lambda x: len([i for i in str(x).split(";") if i.strip()]) if pd.notna(x) and x else 0
    )

    # --- Sponsor features ---
    df["SponsorClass"] = df["LeadSponsorClass"].fillna("OTHER")
    df["IsIndustrySponsored"] = df["SponsorClass"].str.contains("INDUSTRY|Industry", case=False, na=False).astype(int)
    df["IsNIHSponsored"] = df["SponsorClass"].str.contains("NIH", case=False, na=False).astype(int)
    df["IsAcademicSponsored"] = df["SponsorClass"].str.contains(
        "FED|OTHER_GOV|U.S. Fed", case=False, na=False
    ).astype(int)

    # --- Geographic features ---
    df["LocationCountry"] = df["LocationCountry"].fillna("")
    df["NumCountries"] = df["LocationCountry"].apply(
        lambda x: len([c for c in str(x).split(";") if c.strip()]) if pd.notna(x) and x else 0
    )
    df["IsMultiCountry"] = (df["NumCountries"] > 1).astype(int)
    df["IsUSOnly"] = df["LocationCountry"].str.contains("United States", case=False, na=False).astype(int)

    # --- Eligibility features ---
    # Age groups
    df["IncludesChildren"] = df["MinimumAge"].fillna("").str.contains(
        "Year|Month|Child", case=False, regex=True
    ).astype(int)
    df["IncludesAdults"] = (
        df["MaximumAge"].fillna("").str.contains("Year", case=False, na=False) |
        df["MaximumAge"].fillna("").str.contains("N/A|No Limit", case=False, na=False)
    ).astype(int)

    # Gender
    df["AllGenders"] = df["Gender"].fillna("").str.upper().isin(["ALL", ""]).astype(int)
    df["MaleOnly"] = df["Gender"].fillna("").str.upper() == "MALE"
    df["FemaleOnly"] = df["Gender"].fillna("").str.upper() == "FEMALE"

    # Healthy volunteers
    df["AcceptsHealthyVolunteers"] = df["HealthyVolunteers"].astype(str).fillna("").str.contains(
        "Yes|Accepts", case=False, regex=True
    ).astype(int)

    # --- Purpose features ---
    df["IsTreatmentPurpose"] = df["DesignPrimaryPurpose"].fillna("").str.contains(
        "TREATMENT|Treatment", case=False, regex=True
    ).astype(int)
    df["IsPreventionPurpose"] = df["DesignPrimaryPurpose"].fillna("").str.contains(
        "PREVENTION|Prevention", case=False, regex=True
    ).astype(int)

    print(f"Created {len([c for c in df.columns if c not in ['NCTId', 'Label', 'OverallStatus']])} features")

    return df


def select_modeling_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Select final features for modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Data with all engineered features

    Returns
    -------
    Tuple[pd.DataFrame, list]
        Processed dataframe and list of feature names
    """
    # Define feature columns for modeling
    feature_cols = [
        # Enrollment
        "EnrollmentCount",
        "LogEnrollment",
        "SmallTrial",
        "LargeTrial",
        # Phase
        "IsPhase1",
        "IsPhase2",
        "IsPhase3",
        "IsPhase2And3",
        # Design
        "IsRandomized",
        "IsDoubleBlind",
        "IsBlinded",
        "NumberOfArms",
        "HasMultipleArms",
        "NumArms_2",
        "NumArms_3Plus",
        "IsParallelAssignment",
        "IsCrossover",
        # Intervention
        "IsDrugIntervention",
        "IsBehavioralIntervention",
        "IsDeviceIntervention",
        "NumInterventions",
        # Sponsor
        "IsIndustrySponsored",
        "IsNIHSponsored",
        "IsAcademicSponsored",
        # Geography
        "NumCountries",
        "IsMultiCountry",
        "IsUSOnly",
        # Eligibility
        "IncludesChildren",
        "IncludesAdults",
        "AllGenders",
        "AcceptsHealthyVolunteers",
        # Purpose
        "IsTreatmentPurpose",
        "IsPreventionPurpose",
    ]

    # Keep ID, label, and status for reference
    metadata_cols = ["NCTId", "Label", "OverallStatus", "BriefTitle"]

    # Select columns
    selected_cols = metadata_cols + feature_cols
    df_final = df[selected_cols].copy()

    # Handle missing values in features
    # For numeric features, fill with median
    numeric_features = ["EnrollmentCount", "LogEnrollment", "NumberOfArms", "NumInterventions", "NumCountries"]
    for col in numeric_features:
        if col in df_final.columns:
            median_val = df_final[col].median()
            df_final[col] = df_final[col].fillna(median_val)

    # For binary features, fill with 0 (most conservative)
    binary_features = [c for c in feature_cols if c not in numeric_features]
    for col in binary_features:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print(f"{'='*60}")
    print(f"Total samples: {len(df_final)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Missing values per feature:")
    missing = df_final[feature_cols].isnull().sum()
    if missing.sum() == 0:
        print("  No missing values!")
    else:
        print(missing[missing > 0])

    return df_final, feature_cols


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/adhd_trials_labeled.csv"):
    """
    Save processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Processed trial data
    output_path : str
        Path to save processed data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")


def main():
    """Main execution function."""
    print("="*60)
    print("ADHD Clinical Trials Data Preparation")
    print("="*60)

    # Load raw data
    df = load_raw_data()

    # Create labels
    df_labeled = create_binary_labels(df)

    # Engineer features
    df_features = engineer_features(df_labeled)

    # Select final features
    df_final, feature_cols = select_modeling_features(df_features)

    # Save processed data
    save_processed_data(df_final)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)

    return df_final, feature_cols


if __name__ == "__main__":
    main()