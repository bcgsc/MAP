"""
CytEx 1.0
"""
import argparse
import os
import pandas as pd
import numpy as np
from progress.bar import FillingCirclesBar


def excel_file(input_path):
    """
    Validate the file format for "raw_data" and "matrix".
    """
    _, ext = os.path.splitext(input_path)
    if ext not in ['.xlsx', 'xls']:
        raise argparse.ArgumentTypeError(f"{input_path} is not an Excel file")
    return input_path
parser = argparse.ArgumentParser(description="Command line tool for acquiring user input")
parser.add_argument("-a", "--assay", help="Assay type", type=str, choices=['ast', 'hc50'], required=False, default='hc50')
parser.add_argument("-p", "--prefix", help = "Prefix", type = lambda s: s.replace(" ", "-"), required = False, default="Sample_name")
parser.add_argument("-d", "--raw_data", help = "Plate reader data", type = excel_file, required = True)
parser.add_argument("-m", "--matrix", help = "Matrix map", type = excel_file, required = True)
parser.add_argument("-r", "--num_tech_rep", help = "Number of replicates",type = int, required= False, default = 2)
parser.add_argument("-s", "--start_con", help = "Starting concentration", type = int, required = False, default = 128)
parser.add_argument("-f", "--final_con", help = "Final concentration", type = int, required = False, default = 1)
parser.add_argument("-t", "--threshold", help = "MIC threshold", type = int, required = False, default = 0.4999)
args = parser.parse_args()

def detect_plate(row, col, raw_data_array):
    """Detect the start of a new plate using the markers "A" and 1."""
    cell = raw_data_array[row, col]
    cell_south = raw_data_array[row + 1, col]
    cell_east = raw_data_array[row, col + 1]
    return pd.isna(cell) and cell_south == "A" and cell_east == 1

def process_plate(row, col, raw_data_array, matrix_data, master_dict, plate_num):
    """Process a detected plate to capture absorbance value, synth_id, and amc."""
    master_dict[plate_num] = {'wells': {}}
    for i in range(8):
        for j in range(12):
            well = chr(ord('A') + i) + str(j + 1)
            master_dict[plate_num]['wells'][well] = {'abs_val': raw_data_array[row + 1 + i, col + 1 + j]}
            matrix_row = matrix_data[matrix_data['well'] == well]
            if not matrix_row.empty:
                master_dict[plate_num]['wells'][well]['synth_id'] = matrix_row['synth_id'].values[0]
                master_dict[plate_num]['wells'][well]['amc'] = matrix_row['amc'].values[0]

def process_data(raw_data_path, matrix_path):
    """Process the raw data and return a master dictionary with processed plate data."""
    raw_data = pd.read_excel(raw_data_path)
    matrix_data = pd.read_excel(matrix_path)
    raw_data_array = raw_data.to_numpy()
    num_rows, num_cols = raw_data_array.shape
    num_plates = 0
    master_dict = {'plates': {}}
    total_cells = (num_rows - 1) * (num_cols - 1)
    progress_bar = FillingCirclesBar('Processing data:    ', max=total_cells)
    for row in range(num_rows - 1):
        for col in range(num_cols - 1):
            if detect_plate(row, col, raw_data_array):
                num_plates += 1
                process_plate(row, col, raw_data_array, matrix_data, master_dict['plates'], num_plates)
            progress_bar.next()
    progress_bar.finish()
    return master_dict, num_plates

def calculate_concentrations(start_con: float, final_con: float, plates_per_rep: int) -> list:
    """Calculate Concentrations based user input"""
    conc_list = np.logspace(np.log10(start_con),
                                  np.log10(final_con),
                                  plates_per_rep)
    # Apply high-precision rounding and remove unnecessary decimals
    multiplier = 10**10
    conc_list = [np.round(val * multiplier) / multiplier for val in conc_list]
    conc_list = [int(val) if np.isclose(val, int(val)) else val for val in conc_list]
    return conc_list

def validate_input(num_tech_rep: float, start_con: float, final_con: float, num_plates: int, conc_list: list) -> bool:
    """Validate Experimental Setup"""
    if start_con <= final_con:
        print("Starting concentration cannot be smaller than final concentration.")
        return False
    if num_tech_rep > num_plates:
        print("Number of technical replicates cannot exceed the number of plates.")
        return False
    if num_plates % num_tech_rep != 0:
        print("Number of plates is not divisible by the number of technical replicates.")
        return False
    for i in range(len(conc_list)-1):
        ratio = conc_list[i] / conc_list[i+1]
        if not (np.isclose(ratio, 2) or np.isclose(ratio, 5) or np.isclose(ratio, 10)):
            print("The concentrations do not follow a 1:2, 1:5, or 1:10 dilution pattern.")
            return False
    return True

def populate_rep_conc(master_dict: dict, plates_per_rep: int, conc_list: list) -> None:
    """Populate the master_dict with technical replicate identifiers (tech_rep) and concentration values (conc_val)."""
    num_plates = len(master_dict['plates'])
    for plate_index in range(num_plates):
        tech_rep = (plate_index // plates_per_rep) + 1
        conc_val = conc_list[plate_index % plates_per_rep]
        master_dict['plates'][plate_index + 1]['tech_rep'] = tech_rep
        master_dict['plates'][plate_index + 1]['conc_val'] = conc_val

def determine_mic(master_dict: dict):
    """Determine the Minimum Inhibitory Concentration (MIC) for each well in the master dictionary."""
    tech_reps = {plate_data['tech_rep'] for plate_data in master_dict['plates'].values()}
    total_wells = len(next(iter(master_dict['plates'].values()))['wells'])
    total_tech_reps = len(tech_reps)
    progress_bar = FillingCirclesBar('Determining MIC:    ', max=total_tech_reps * total_wells)
    for tech_rep in tech_reps:
        plates_in_rep = [plate_data
                         for plate_data in master_dict['plates'].values()
                         if plate_data['tech_rep'] == tech_rep]
        for well in plates_in_rep[0]['wells']:
            mic_value = None
            for plate_data in sorted(plates_in_rep, key=lambda x: x['conc_val']):
                abs_val = plate_data['wells'][well]['abs_val']
                amc = plate_data['wells'][well].get('amc')
                # Handle Sterility Control and Growth Control
                if amc in ["Sterility Control", "Growth Control"]:
                    mic_value = "N/A"
                    break
                threshold = args.threshold
                if abs_val <= threshold:
                    mic_value = plate_data['conc_val']
                    break
            if mic_value is None:
                highest_concentration = sorted(plates_in_rep, key=lambda x: -x['conc_val'])[0]['conc_val']
                mic_value = f">{highest_concentration}"
            # Update master_dict with MIC values
            for plate_data in plates_in_rep:
                plate_data['wells'][well]['mic'] = mic_value
            progress_bar.next()
    progress_bar.finish()

def determine_hc50(master_dict: dict):
    """Determine the Hemolytic activity (HC50) for each well in the master dictionary."""
    tech_reps = {plate_data['tech_rep'] for plate_data in master_dict['plates'].values()}
    total_wells = len(next(iter(master_dict['plates'].values()))['wells'])
    total_tech_reps = len(tech_reps)
    progress_bar = FillingCirclesBar('Determining HC50:   ', max=total_tech_reps * total_wells)
    for tech_rep in tech_reps:
        plates_in_rep = [plate_data
                         for plate_data in master_dict['plates'].values() if plate_data['tech_rep'] == tech_rep]
        plates_in_rep = sorted(plates_in_rep, key=lambda x: x['conc_val'])
        positive_controls = [well_data['abs_val']
                             for plate_data in plates_in_rep
                             for well_data in plate_data['wells'].values()
                             if well_data.get('amc') == "Growth Control"]
        negative_controls = [well_data['abs_val']
                             for plate_data in plates_in_rep
                             for well_data in plate_data['wells'].values()
                             if well_data.get('amc') == "Sterility Control"]
        avg_positive = sum(positive_controls) / len(positive_controls) if positive_controls else 0
        avg_negative = sum(negative_controls) / len(negative_controls) if negative_controls else 0
        # Calculate HC50 threshold
        absorbance_range = avg_positive - avg_negative
        absorbance_at_hc50 = avg_negative + (0.50 * absorbance_range)
        # Update master_dict at with HC50 threshold
        for plate_data in plates_in_rep:
            plate_data['abs_at_hc50'] = absorbance_at_hc50
        for well in plates_in_rep[0]['wells']:
            amc_value = plates_in_rep[0]['wells'][well].get('amc')
            if amc_value in ["Growth Control", "Sterility Control"]:
                hc50_value = "N/A"
            else:
                hc50_value = next((plate_data['conc_val']
                                   for plate_data in plates_in_rep
                                   if plate_data['wells'][well]['abs_val'] >= absorbance_at_hc50), None)
                if hc50_value is None:
                    hc50_value = f">{plates_in_rep[-1]['conc_val']}"
            # Update master_dict with HC50 values
            for plate_data in plates_in_rep:
                plate_data['wells'][well]['hc50'] = hc50_value
            progress_bar.next()
    progress_bar.finish()

def compute_selected_assay(assay: str, master_dict: dict):
    """Compute the selected assay metric (MIC or HC50) and update the master_dict"""
    if assay == 'ast':
        determine_mic(master_dict)
    elif assay == 'hc50':
        determine_hc50(master_dict)

def main():
    """
    main
    """
    # Process the raw data
    master_dict, num_plates = process_data(args.raw_data, args.matrix)
    # Caclulate number of plates per technical replicate group
    plates_per_rep = num_plates // args.num_tech_rep
    # Calculate concetration based on user input
    conc_list = calculate_concentrations(args.start_con, args.final_con, plates_per_rep)
    # Validate the inputs
    valid = validate_input(args.num_tech_rep, args.start_con, args.final_con, num_plates, conc_list)
    if not valid:
        return
    # Populating plates in master_dict with tech_rep and corresponding conc-val.
    populate_rep_conc(master_dict, plates_per_rep, conc_list)
    # Compute MIC or HC50 based on user selection
    compute_selected_assay(args.assay, master_dict)

if __name__ == "__main__":
    main()
