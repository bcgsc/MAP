<p align="center">
  <img width="100" alt="MAP-Logo" src=/projects/amp/asalehi/Git/CytEx/logo/MAP-Logo.png>
</p>

# Microdilution Assay Processor (MAP)
MAP automates data processing and analysis for high-throughput and manual microdilution assays, efficiently calculating Minimum Inhibitory Concentration (MIC), Half-Maximal Hemolytic Concentration (HC50), and Half-Maximal Cytotoxic Concentration (CC50), and generating comprehensive experimental reports.

# Table of Contents
1. [Microdilution Assay Processor (MAP)](#microdilution-assay-processor-map)
2. [Installation](#installation)
3. [Dependencies](#dependencies)
4. [Data Structure](#data-structure)
5. [Input](#input)
6. [Assay Analysis](#assay-analysis)
7. [Usage](#usage)
8. [Example Command](#example-command)

# Installation
## Clone the Repository
To clone the MAP repository, use the following command:
```
git clone https://github.com/bcgsc/MAP.git
cd MAP
```

## Create a Conda Environment
First, create a new Conda environment:
```
conda create --name map python=3.10.12
conda activate map
```
## Install MAP
Once the environment is activated, install MAP using the following command at the root directory.:
```
pip install .
map --version
```
## Testing Installation
To ensure MAP works as intended, run the test files as follows:

```
chmod +x run_tests.sh
./run_tests.sh
```

The test results will be saved in the `tests/test_results` directory. Compare the contents of this directory with the expected results in the `tests/expected_results` directory to verify that MAP worked as expected.

# Dependencies
MAP requires the following dependencies that are installed during installation:
* python 3.10.12+
* pandas
* numpy
* progress
* openpyxl
* xlsxwriter

# Data Structure
MAP organizes its data using a hierarchical dictionary structure. This structure supports scalability and flexibility, allowing easy addition of plates, wells, or assay types. It reduces redundancy, speeds up data retrieval, and clearly defines relationships for analysis.

### MAP Dictionary:
- **Plates**: The top-level dictionary containing all the plates processed in the assay.
  - **Plate**: Each plate is uniquely identified within the Plates dictionary.
    - **Wells**: A dictionary containing information about each well in the plate.
      - **Well**: The key is the well identifier (e.g., “A1”, “B2”).
        - **Absorbance Value**: The absorbance reading for that specific well.
        - **Synthesis ID**: An identifier for the synthesized compound tested in that well.
        - **Antimicrobial Compound Name (AMC)**: The name of the antimicrobial compound tested.
        - **Concentration Value**: The concentration of the compound.
        - **MIC, HC50, or CC50 value**: The determined value for MIC (Minimum Inhibitory Concentration), HC50 (Half-Maximal Hemolytic Concentration), or CC50 (Half-Maximal Cytotoxic Concentration), depending on the assay.
        - **Technical Replicate Number**: Identifies the replicate group for well
- **Absorbance Threshold**: The provided threshold value for AST or the calculated threshold for HC50 or CC50 assays.

# Input
MAP requires two main input files in Excel format: raw data and matrix. Ensure your input files match the provided requirements. For reference templates, please check the `template` directory.

## Raw Data
For both manual and high-throughput assays, the Excel file must contain plate read data in the form of 96-well plates, where:
- 96-well plates must have column headers (1-12) and row identifiers (A-H).

<p align="center">
  <img width="800" alt=Raw-Data" src=/projects/amp/asalehi/Git/CytEx/logo/Manual-vs-Hts.png>
</p>

### High-Throughput:
In high-throughput assays, concentration changes across plates are expected:
- The first plate in the raw data should have the highest concentration, and the last plate should have the lowest concentration.
  - For assays with technical replicates, the replicate groups should appear sequentially. Within each group, plates should be ordered from highest to lowest concentration.

### Manual:
In manual assays, concentration changes across rows where the concentration decrease from left to right in each row.

## Matrix
### High-Throughput:
A single sheet with the following columns:
- **well**: Specifies the well location (e.g., A1, B12).
- **synth_id**: The synthesis identifier for the compound in the well.
- **amc**: The antimicrobial compound name or treatment name.

### Manual:
A single sheet with the following columns:
- **well**: Specifies the well range in a horizontal or vertical array of wells:
  - In horizontal ranges, the row identifier remains unchanged and only the column header changes (e.g., A1-A12).
  - In vertical ranges, the column header remains unchanged and only the row identifier changes (e.g., A12-H12).
- **synth_id**: The synthesis identifier for the compound in the well.
- **amc**: The antimicrobial compound name or treatment name.


# Assay Analysis
Detailed methodologies for calculating MIC, CC50, and HC50 can be found in the `methods` directory.


# Usage

```plaintext
usage: map [-h] [--version] -a {ast,hc50,cc50} -o {manual,hts} [-p PREFIX] -d RAW_DATA -m MATRIX [-r NUM_TECH_REP] -s START_CON -f
           FINAL_CON [-t THRESHOLD] [-e OUTPUT_DIR]

MAP: Microdilution Assay Processor

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -a {ast,hc50,cc50}, --assay {ast,hc50,cc50}
                        Assay type, antimicrobial susceptibility testing, hemolysis assay, or AlamarBlue assay
  -o {manual,hts}, --operation_mode {manual,hts}
                        Operation mode, manual or high-throughput
  -p PREFIX, --prefix PREFIX
                        Prefix for the output Excel file
  -d RAW_DATA, --raw_data RAW_DATA
                        Excel file containing the raw plate reader data in format of 96-well plates
  -m MATRIX, --matrix MATRIX
                        Excel file containing the matrix map, specifying the treatment name (amc) and ID (Synth ID)
  -r NUM_TECH_REP, --num_tech_rep NUM_TECH_REP
                        Number of technical replicates used in high-throughput assay
  -s START_CON, --start_con START_CON
                        Starting concentration (highest concentration) of the assay
  -f FINAL_CON, --final_con FINAL_CON
                        Final concentration (lowest concentration) of the assay
  -t THRESHOLD, --threshold THRESHOLD
                        Absorbance threshold for determination of minimum inhibitory concentration (Default = 0.4999)
  -e OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Directory for the output Excel file
```

# Example Command
```
map -a ast -o hts -p E.coli_ATCC_25922_Hts -d path/to/data.xlsx -m path/to/matrix.xlsx -r 2 -s 128 -f 1 -t 0.49 -e path/to/output
```
This command runs the MAP with the following options:
- **-a ast:** Specifies the assay type as Antimicrobial Susceptibility Testing (AST).
- **-o hts:** Sets the operation mode to high-throughput .
- **-p demo:** Prefixes the output file with "E.coli_ATCC_25922_Hts".
- **-d path/to/data.xlsx:** Path to the raw data Excel file.
- **-m path/to/matrix.xlsx:** Path to the matrix Excel file.
- **-r 2:** Number of technical replicates.
- **-s 128:** Starting concentration.
- **-f 1:** Final concentration.
- **-t 0.4999:** Absorbance threshold.
- **-e path/to/output:** Directory for the output Excel file.