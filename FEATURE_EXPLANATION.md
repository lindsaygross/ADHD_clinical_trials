# ADHD Clinical Trial Feature Explanation

This document provides a detailed explanation of the features used in the machine learning models to predict the success
of ADHD clinical trials. It also defines key clinical trial terminology used in this project.

## Key Terminology

### 1. Arm

In a clinical trial, an **arm** is a group of participants that receives a specific intervention (or no
intervention/placebo) according to the study protocol.

- **Experimental Arm**: Participants receive the new treatment being tested.
- **Control Arm**: Participants receive a placebo, standard of care, or no treatment for comparison.
- **Multi-arm Trial**: A study with more than two arms (e.g., Low Dose, High Dose, Placebo).

### 2. Intervention

An **intervention** is a process or action that is the focus of a clinical study. It can be a:

- **Drug**: Investigational new drug or approved drug used in a new way.
- **Behavioral Therapy**: Psychotherapy, cognitive training, lifestyle changes.
- **Device**: Medical device (e.g., neurofeedback headset).
- **Procedure**: Surgery or diagnostic test.

---

## Feature Definitions

The following features are engineered from the raw ClinicalTrials.gov data. They represent information available **at
the start** of the trial, ensuring no data leakage.

### Enrollment Features

These features relate to the number of participants in the study.

| Feature Name        | Description                                       | Logic / Meaning                                                               |
|:--------------------|:--------------------------------------------------|:------------------------------------------------------------------------------|
| **EnrollmentCount** | The target number of participants to be enrolled. | Extracted directly from the protocol. Missing values imputed with median.     |
| **LogEnrollment**   | Log-transformed enrollment count.                 | $ln(EnrollmentCount + 1)$. Used to handle skewed distribution of trial sizes. |
| **SmallTrial**      | Binary indicator for small studies.               | 1 if Enrollment < 50, else 0.                                                 |
| **LargeTrial**      | Binary indicator for large studies.               | 1 if Enrollment ≥ 200, else 0.                                                |

### Phase Features

Clinical trials are conducted in phases.

| Feature Name     | Description                        | Logic / Meaning                                    |
|:-----------------|:-----------------------------------|:---------------------------------------------------|
| **IsPhase1**     | Early-stage safety trials.         | 1 if the trial is explicitly Phase 1.              |
| **IsPhase2**     | Safety and efficacy trials.        | 1 if the trial involves Phase 2.                   |
| **IsPhase3**     | Large-scale efficacy confirmation. | 1 if the trial involves Phase 3.                   |
| **IsPhase2And3** | Combined phase trials.             | 1 if the trial is designated as "Phase 2/Phase 3". |

### Design Features

Features describing how the study is structured to ensure scientific validity.

| Feature Name             | Description                                                       | Logic / Meaning                                                              |
|:-------------------------|:------------------------------------------------------------------|:-----------------------------------------------------------------------------|
| **IsRandomized**         | Participants are randomly assigned to groups.                     | 1 if allocation includes "Randomized". Reduces selection bias.               |
| **IsDoubleBlind**        | Neither participants nor researchers know who gets the treatment. | 1 if masking is "Double" or "Triple". High-quality design standard.          |
| **IsBlinded**            | At least one party is blinded.                                    | 1 if masking is not "None" or "Open Label".                                  |
| **NumberOfArms**         | Total count of arms in the study.                                 | Numeric count. 2 is standard (Treatment vs Control).                         |
| **HasMultipleArms**      | Indicator for > 1 arm.                                            | 1 if NumberOfArms > 1. Single-arm studies are often open-label/early phase.  |
| **NumArms_2**            | Exactly two arms.                                                 | 1 if NumberOfArms == 2.                                                      |
| **NumArms_3Plus**        | Three or more arms.                                               | 1 if NumberOfArms ≥ 3.                                                       |
| **IsParallelAssignment** | Participants stay in one arm throughout.                          | 1 if intervention model is "Parallel". Most common design.                   |
| **IsCrossover**          | Participants switch treatments during the study.                  | 1 if intervention model is "Crossover". Subjects serve as their own control. |

### Intervention Features

The type of treatment being studied.

| Feature Name                 | Description                               | Logic / Meaning                                     |
|:-----------------------------|:------------------------------------------|:----------------------------------------------------|
| **IsDrugIntervention**       | Study involves a pharmaceutical drug.     | 1 if intervention type includes "Drug".             |
| **IsBehavioralIntervention** | Study involves therapy/behavioral change. | 1 if intervention type includes "Behavioral".       |
| **IsDeviceIntervention**     | Study involves a medical device.          | 1 if intervention type includes "Device".           |
| **NumInterventions**         | Count of distinct interventions listed.   | Number of items listed in the intervention section. |

### Sponsor Features

Who is funding and leading the study.

| Feature Name            | Description                                   | Logic / Meaning                                                                      |
|:------------------------|:----------------------------------------------|:-------------------------------------------------------------------------------------|
| **IsIndustrySponsored** | Funded by a pharmaceutical/biotech company.   | 1 if sponsor class is "INDUSTRY". Often has more resources but different incentives. |
| **IsNIHSponsored**      | Funded by the National Institutes of Health.  | 1 if sponsor class is "NIH".                                                         |
| **IsAcademicSponsored** | Funded by university or government (non-NIH). | 1 if sponsor class is "OTHER_GOV" (Fed) or similar academic/federal sources.         |

### Geographic Features

Where the study is taking place.

| Feature Name       | Description                   | Logic / Meaning                                                   |
|:-------------------|:------------------------------|:------------------------------------------------------------------|
| **NumCountries**   | Number of countries involved. | Count of unique countries in the locations list.                  |
| **IsMultiCountry** | International study.          | 1 if NumCountries > 1. Indicates higher complexity and resources. |
| **IsUSOnly**       | Domestic US study.            | 1 if the only country listed is "United States".                  |

### Eligibility Features

Criteria for who can join the study.

| Feature Name                 | Description                    | Logic / Meaning                                                                   |
|:-----------------------------|:-------------------------------|:----------------------------------------------------------------------------------|
| **IncludesChildren**         | Allows pediatric participants. | 1 if minimum age indicates months/years/child.                                    |
| **IncludesAdults**           | Allows adult participants.     | 1 if maximum age allows adults or is "N/A".                                       |
| **AllGenders**               | Open to all sexes.             | 1 if gender is not restricted to Male or Female only.                             |
| **AcceptsHealthyVolunteers** | Allows people without ADHD.    | 1 if healthy volunteers status is "Yes" or "Accepts". Usually for Phase 1 safety. |

### Purpose Features

The primary scientific goal of the study.

| Feature Name            | Description                       | Logic / Meaning                       |
|:------------------------|:----------------------------------|:--------------------------------------|
| **IsTreatmentPurpose**  | Goal is to treat the condition.   | 1 if primary purpose is "Treatment".  |
| **IsPreventionPurpose** | Goal is to prevent the condition. | 1 if primary purpose is "Prevention". |
