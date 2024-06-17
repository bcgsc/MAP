"""
MAP 1.0.1
"""
import argparse
import os
from datetime import datetime
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
parser = argparse.ArgumentParser(description="MAP:  Microdilution Assay Processor ")
parser.add_argument('--version', action='version', version='MAP 1.0.1')
parser.add_argument("-a", "--assay", help ="Assay type, antimicrobial susceptibility testing, hemolysis assay, or AlamarBlue assay", type=str, choices=['ast', 'hc50','cc50'], required = True)
parser.add_argument("-o", "--operation_mode", help ="Operation mode, manual or high-throughput", type=str, choices=['manual', 'hts'], required = True)
parser.add_argument("-p", "--prefix", help = "Prefix for the output Excel file", type = lambda s: s.replace(" ", "-"), required = False, default="Sample_name")
parser.add_argument("-d", "--raw_data", help = "Excel file contaning the raw plate reader data in format of 96-well plates", type = excel_file, required = True)
parser.add_argument("-m", "--matrix", help = "Excel file contaning the matrix map, specifying the treatment name (amc) and ID (Synth ID)", type = excel_file, required = True)
parser.add_argument("-r", "--num_tech_rep", help = "Number of technical replicates used in high-throughput assay",type = int, required= False, default = 2)
parser.add_argument("-s", "--start_con", help = "Starting concentration (highest concentration) of the assay", type = float, required = True)
parser.add_argument("-f", "--final_con", help = "Final concentration (lowest concentration) of the assay", type = float, required = True)
parser.add_argument("-t", "--threshold", help = "Absorbance threshold for determination of minimum inhibitory concetration (Default = 0.4999)", type = float, required = False, default = 0.4999)
parser.add_argument("-e", "--output_dir", help = "Directory for the output Excel file", type = str, required = False, default = "results")
args = parser.parse_args()

def detect_plate(row, col, raw_data_array):
    """Detect the start of a new plate using the markers "A" and 1."""
    cell = raw_data_array[row, col]
    cell_south = raw_data_array[row + 1, col]
    cell_east = raw_data_array[row, col + 1]
    return pd.isna(cell) and cell_south == "A" and cell_east == 1

def expand_well_ranges(matrix_data):
    """Expand ranges of wells for the manual matrix map"""
    new_rows = []
    for _, row in matrix_data.iterrows():
        well_range = row['well'].split('-')
        if len(well_range) == 2:
            start_well, end_well = well_range
            start_row, start_col = ord(start_well[0]), int(start_well[1:])
            end_row, end_col = ord(end_well[0]), int(end_well[1:])
            if start_row == end_row:
                for col in range(start_col, end_col + 1):
                    new_row = row.copy()
                    new_row['well'] = f"{chr(start_row)}{col}"
                    new_rows.append(new_row)
            else:
                if start_col == end_col:
                    for row_char in range(start_row, end_row + 1):
                        new_row = row.copy()
                        new_row['well'] = f"{chr(row_char)}{start_col}"
                        new_rows.append(new_row)
                else:
                    for row_char in range(start_row, end_row + 1):
                        if row_char == start_row:
                            col_start, col_end = start_col, 13
                        elif row_char == end_row:
                            col_start, col_end = 1, end_col + 1
                        else:
                            col_start, col_end = 1, 13
                        for col in range(col_start, col_end):
                            new_row = row.copy()
                            new_row['well'] = f"{chr(row_char)}{col}"
                            new_rows.append(new_row)
        else:
            new_rows.append(row)
    return pd.DataFrame(new_rows)



def process_plate(row, col, raw_data_array, matrix_data, master_dict, plate_num, operation_mode):
    """
    Process the detected plate to capture absorbance value, synth_id, and amc.
    """
    if operation_mode == 'manual':
        master_dict[plate_num] = {'rows': {}}
        for i in range(8):
            row_id = chr(ord('A') + i)
            master_dict[plate_num]['rows'][row_id] = {'wells': {}}
            for j in range(12):
                well = chr(ord('A') + i) + str(j + 1)
                abs_val = raw_data_array[row + 1 + i, col + 1 + j]
                matrix_row = matrix_data[(matrix_data['well'] == well) & (matrix_data['plate_number'] == plate_num)]
                master_dict[plate_num]['rows'][row_id]['wells'][well] = {'abs_val': abs_val}
                if not matrix_row.empty:
                    master_dict[plate_num]['rows'][row_id]['wells'][well]['synth_id'] = matrix_row['synth_id'].values[0]
                    master_dict[plate_num]['rows'][row_id]['wells'][well]['amc'] = matrix_row['amc'].values[0]
    else:
        master_dict[plate_num] = {'wells': {}}
        for i in range(8):
            for j in range(12):
                well = chr(ord('A') + i) + str(j + 1)
                master_dict[plate_num]['wells'][well] = {'abs_val': raw_data_array[row + 1 + i, col + 1 + j]}
                matrix_row = matrix_data[matrix_data['well'] == well]
                if not matrix_row.empty:
                    master_dict[plate_num]['wells'][well]['synth_id'] = matrix_row['synth_id'].values[0]
                    master_dict[plate_num]['wells'][well]['amc'] = matrix_row['amc'].values[0]

def process_data(raw_data_path, matrix_path, operation_mode):
    """Process the raw data and return a master dictionary with processed plate data."""
    raw_data = pd.read_excel(raw_data_path)
    matrix_data = pd.read_excel(matrix_path)
    if operation_mode == 'manual':
        matrix_data = expand_well_ranges(matrix_data)
    raw_data_array = raw_data.to_numpy()
    num_rows, num_cols = raw_data_array.shape
    num_plates = 0
    master_dict = {'plates': {}}
    total_cells = (num_rows - 1) * (num_cols - 1)
    progress_bar = FillingCirclesBar('Processing data:      ', max=total_cells)
    for row in range(num_rows - 1):
        for col in range(num_cols - 1):
            if detect_plate(row, col, raw_data_array):
                num_plates += 1
                process_plate(row, col, raw_data_array, matrix_data, master_dict['plates'], num_plates, operation_mode)
            progress_bar.next()
    progress_bar.finish()
    return master_dict, num_plates

def calculate_concentrations(start_con, final_con, plates_per_rep, operation_mode, master_dict):
    """Calculate Concentrations based on user input."""
    if args.operation_mode == 'manual':
        wells = master_dict['plates'][1]['rows']['A']['wells'].values()
        wells_per_treatment  = sum(details.get('amc') not in ["Growth Control", "Sterility Control"] for details in wells)
    num_points = plates_per_rep if operation_mode == 'hts' else wells_per_treatment
    conc_list = np.logspace(np.log2(start_con), np.log2(final_con), num_points, base=2)
    conc_list = np.around(conc_list, decimals=3)
    conc_list = [int(val) if np.isclose(val, int(val)) else val for val in conc_list]
    return conc_list

def validate_input(num_tech_rep, start_con, final_con, num_plates, conc_list, operation_mode):
    """Validate Experimental Setup"""
    if operation_mode == 'hts':
        if num_tech_rep > num_plates:
            print("Number of technical replicates cannot exceed the number of plates.")
            return False
        if num_plates % num_tech_rep != 0:
            print("Number of plates is not divisible by the number of technical replicates.")
            return False
    if start_con <= final_con:
        print("Starting concentration cannot be smaller than final concentration.")
        return False
    for i, j in zip(conc_list, conc_list[1:]):
        ratio = i / j
        if not (np.isclose(ratio, 2) or np.isclose(ratio, 5) or np.isclose(ratio, 10)):
            print("The concentrations do not follow a 1:2, 1:5, or 1:10 dilution pattern.")
            return False
    return True

def assign_manual_rep(master_dict):
    """Determine technical replicate for each row of the manual plates and populate master_dict."""
    combination_counts = {}
    for plate_data in master_dict['plates'].values():
        for row_data in plate_data['rows'].values():
            first_well = next(iter(row_data['wells'].values()))
            combination_key = (first_well['amc'], first_well['synth_id'])
            tech_rep_num = combination_counts.get(combination_key, 0) + 1
            combination_counts[combination_key] = tech_rep_num
            for well_data in row_data['wells'].values():
                well_data['tech_rep'] = tech_rep_num
    return master_dict

def assign_conc_manual(master_dict, conc_list):
    """ Populates concentrations in the master_dict for manual plates, excluding Sterility Control and Growth Control"""
    conc_idx = 0
    total_conc_assignments = len(conc_list)
    for plate_data in master_dict['plates'].values():
        for row_data in plate_data['rows'].values():
            for well_data in row_data['wells'].values():
                if well_data['amc'] not in ['Sterility Control', 'Growth Control']:
                    well_data['conc_val'] = conc_list[conc_idx % total_conc_assignments]
                    conc_idx += 1  # Move to the next concentration
    return master_dict

def populate_rep_conc_hts(master_dict, plates_per_rep, conc_list):
    """Populate the master_dict with technical replicate identifiers (tech_rep) and concentration values (conc_val)."""
    num_plates = len(master_dict['plates'])
    for plate_number in master_dict['plates']:
        tech_rep = ((plate_number - 1) // plates_per_rep) + 1
        conc_val = conc_list[(plate_number - 1) % plates_per_rep]
        master_dict['plates'][plate_number]['tech_rep'] = tech_rep
        master_dict['plates'][plate_number]['conc_val'] = conc_val

def add_notes(concentrations_above_threshold):
    """Constructs a note for wells with absorbance values above the threshold after the MIC well."""
    if concentrations_above_threshold:
        concentration_word = "concentration" if len(concentrations_above_threshold) == 1 else "concentrations"
        concs = ", ".join(map(str, concentrations_above_threshold))
        return f"At {concentration_word} {concs}, absorbance values greater than the threshold were observed."
    else:
        return ""

def determine_mic_manual(master_dict):
    """Determine MIC for manual operation mode"""
    total_wells = sum(len([data
                           for data in row['wells'].values()
                           if data.get('amc') not in ["Sterility Control", "Growth Control"]])
                           for plate in master_dict['plates'].values()
                           for row in plate['rows'].values())
    progress_bar = FillingCirclesBar('Determining MIC:      ', max=total_wells)
    for plate_num, plate_data in master_dict['plates'].items():
        for row_id, row_data in plate_data['rows'].items():
            filtered_wells = {
                well: data for well, data in row_data['wells'].items()
                if 'conc_val' in data and data.get('amc') not in ["Sterility Control", "Growth Control"]}
            sorted_wells = sorted(filtered_wells.items(), key=lambda x: x[1]['conc_val'])
            mic_value = None
            concentrations_above_threshold = []
            mic_found = False
            for well, data in sorted_wells:
                progress_bar.next()
                if data['abs_val'] <= args.threshold and not mic_found:
                    mic_value = data['conc_val']
                    mic_found = True
                elif mic_found and data['abs_val'] > args.threshold:
                    concentrations_above_threshold.append(data['conc_val'])
            if mic_value is None and sorted_wells:
                highest_concentration = args.start_con
                mic_value = f">{highest_concentration}"
            additional_note = add_notes(concentrations_above_threshold) if mic_value is not None else ""
            for _, data in filtered_wells.items():
                data['mic'] = mic_value
                data.setdefault('note', '')
                if additional_note:
                    data['note'] += (" " + additional_note if data['note'] else additional_note)
    progress_bar.finish()

def determine_mic_hts(master_dict):
    """
    Determine the Minimum Inhibitory Concentration (MIC) for each well in the master dictionary.
    """
    total_tech_reps = args.num_tech_rep
    tech_reps = set(range(1, total_tech_reps + 1))
    total_wells = len(next(iter(master_dict['plates'].values()))['wells'])
    progress_bar = FillingCirclesBar('Determining MIC:      ', max=total_tech_reps * total_wells)
    for tech_rep in tech_reps:
        plates_in_rep = [plate_data
                         for plate_data in master_dict['plates'].values()
                         if plate_data['tech_rep'] == tech_rep]
        for well in plates_in_rep[0]['wells']:
            mic_value = None
            concentrations_above_threshold = []
            for plate_data in sorted(plates_in_rep, key=lambda x: x['conc_val']):
                abs_val = plate_data['wells'][well]['abs_val']
                amc = plate_data['wells'][well].get('amc')
                if amc in ["Sterility Control", "Growth Control"]:
                    mic_value = "N/A"
                    break
                if abs_val <= args.threshold and mic_value is None:
                    mic_value = plate_data['conc_val']
                elif abs_val > args.threshold and mic_value is not None:
                    concentrations_above_threshold.append(plate_data['conc_val'])
            additional_note = add_notes(concentrations_above_threshold)
            if mic_value is None:
                highest_concentration = args.start_con
                mic_value = f">{highest_concentration}"
            for plate_data in plates_in_rep:
                plate_data['wells'][well]['mic'] = mic_value
                if additional_note:
                    plate_data['wells'][well].setdefault('note', '')
                    plate_data['wells'][well]['note'] += (" " + additional_note if plate_data['wells'][well]['note'] else additional_note)
            progress_bar.next()
    progress_bar.finish()

def determine_hc50_cc50_manual(master_dict, assay):
    """Determine the Hemolytic Concentration 50 (HC50) or Cytotoxic Concentration 50 (CC50) for manual operation mode."""
    positive_controls = []
    negative_controls = []
    for plate in master_dict['plates'].values():
        for row in plate['rows'].values():
            for data in row['wells'].values():
                if data.get('amc') == "Growth Control":
                    positive_controls.append(data['abs_val'])
                elif data.get('amc') == "Sterility Control":
                    negative_controls.append(data['abs_val'])
    avg_positive = sum(positive_controls) / len(positive_controls) if positive_controls else 0
    avg_negative = sum(negative_controls) / len(negative_controls) if negative_controls else 0
    absorbance_range = avg_positive - avg_negative if assay == 'hc50' else avg_negative - avg_positive
    non_control_wells = sum(
        1 for plate in master_dict['plates'].values() for row in plate['rows'].values()
        for well in row['wells'].values() if well.get('amc') not in ["Growth Control", "Sterility Control"])
    progress_bar = FillingCirclesBar(f'Determining {assay.upper()}:     ', max=non_control_wells)
    for plate_num, plate_data in master_dict['plates'].items():
        absorbance_at_target = avg_negative + (0.50 * absorbance_range) if assay == 'hc50' else avg_positive + (0.50 * absorbance_range)
        for row_id, row_data in plate_data['rows'].items():
            target_value = None
            sorted_wells = sorted([(well, data) for well, data in row_data['wells'].items() if 'conc_val' in data], key=lambda item: item[1]['conc_val'])
            for well, data in sorted_wells:
                progress_bar.next()
                if data.get('amc') in ["Growth Control", "Sterility Control"]:
                    continue
                data[f'abs_at_{assay}'] = absorbance_at_target
                if ((data['abs_val'] >= absorbance_at_target and assay == 'hc50') or
                   (data['abs_val'] <= absorbance_at_target and assay == 'cc50')) and target_value is None:
                    target_value = data['conc_val']
                    break
            if target_value is not None:
                for well, data in row_data['wells'].items():
                    if data.get('amc') not in ["Growth Control", "Sterility Control"]:
                        data[assay] = target_value
                        progress_bar.next()
            else:
                highest_concentration = args.start_con
                for well, data in row_data['wells'].items():
                    if data.get('amc') not in ["Growth Control", "Sterility Control"]:
                        data[assay] = f">{highest_concentration}"
    progress_bar.finish()


def determine_hc50_hts(master_dict):
    """Determine the Hemolytic activity (HC50) for each well in the master dictionary."""
    total_tech_reps = args.num_tech_rep
    tech_reps = set(range(1, total_tech_reps + 1))
    total_wells = len(next(iter(master_dict['plates'].values()))['wells'])
    progress_bar = FillingCirclesBar('Determining HC50:     ', max=total_tech_reps * total_wells)
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
        absorbance_range = avg_positive - avg_negative
        absorbance_at_hc50 = avg_negative + (0.50 * absorbance_range)
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
            for plate_data in plates_in_rep:
                plate_data['wells'][well]['hc50'] = hc50_value
            progress_bar.next()
    progress_bar.finish()

def compute_selected_assay(assay, master_dict, operation_mode):
    """Compute the selected assay metric (MIC or HC50) and update the master_dict"""
    if assay == 'ast':
        if operation_mode == 'hts':
            determine_mic_hts(master_dict)
        if operation_mode == 'manual':
            determine_mic_manual(master_dict)
    if assay == 'hc50':
        if operation_mode =='hts':
            determine_hc50_hts(master_dict)
        if operation_mode == 'manual':
            determine_hc50_cc50_manual(master_dict, assay)
    if assay == "cc50":
        determine_hc50_cc50_manual(master_dict, assay)

def generate_excel(master_dict, assay, prefix, operation_mode, output_dir):
    """Generate an Excel file output containing different sheets using the master_dict"""
    total_steps = 2 if operation_mode == 'manual' else 3
    progress_bar = FillingCirclesBar('Generating Excel:     ', max=total_steps)
    date = datetime.now().strftime('%Y%m%d')
    file_name = f"{prefix}_{assay.upper()}_{date}"
    counter = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{file_name}.xlsx")
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(output_dir, f"{file_name}_{counter}.xlsx")
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        if operation_mode == 'manual':
            create_visual_data_sheet_manual(master_dict, writer, assay, progress_bar)
            create_mic_c50_sheet_manual(master_dict, writer, assay, progress_bar)
        if operation_mode == 'hts':
            create_visual_data_sheet_hts(master_dict, assay, writer, progress_bar)
            create_mic_hc50_sheet_hts(master_dict, assay, writer, progress_bar)
            create_raw_plate_sheet(master_dict, assay, writer, progress_bar)
    progress_bar.finish()
    print(f"\033[92mExcel output has been saved to {output_path}\033[0m")
def cell_formats(workbook):
    """Initialize and return common cell formats for use in Excel sheets"""
    header_format = workbook.add_format({'bold': True,'align': 'center','valign': 'vcenter','border': 1})
    cell_format = workbook.add_format({'align': 'center','valign': 'vcenter','border': 1})
    return header_format, cell_format

def initialize_worksheet(writer, sheet_name):
    """Initialize Excel worksheet and formating """
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    header_format, cell_format = cell_formats(workbook)
    return workbook, worksheet, header_format, cell_format

def create_visual_data_sheet_manual(master_dict, writer, assay, progress_bar):
    """ Create the visual data Excel sheet for manual assay """
    workbook, worksheet, header_format, cell_format = initialize_worksheet(writer, 'Visual_Data')
    if assay == 'ast':
        column_widths = {'B:B': 6, 'C:N': 6, 'O:O': 14, 'P:P': 9, 'Q:R': 6, 'S:S': 70}
    if assay == 'hc50':
        column_widths = {'B:B': 6, 'C:N': 6, 'O:O': 14, 'P:P': 9, 'Q:Q': 6, 'R:R': 70}
    if assay == 'cc50':
        column_widths = {'B:B': 6, 'C:N': 6, 'O:O': 14, 'P:P': 9, 'Q:Q': 6, 'R:R': 70}
    for columns, width in column_widths.items():
        worksheet.set_column(columns, width)
    row_index = 1
    assay_column_header = "CC50" if assay == "cc50" else "HC50" if assay == "hc50" else "MIC"
    additional_headers = ["Synth_ID", "AMC", assay_column_header] + (["MBC"] if assay == "ast" else []) + ["Notes"]
    for plate_number, plate_data in master_dict['plates'].items():
        column_headers = [''] * 12
        for row_data in plate_data['rows'].values():
            for col_num, (_, well_info) in enumerate(row_data['wells'].items(), start=2):
                key = well_info.get('amc', '')
                column_headers[col_num-2] = '+' if 'Growth Control' in key else '-' if 'Sterility Control' in key else well_info.get('conc_val', 'N/A')
        worksheet.merge_range(row_index, 1, row_index, len(column_headers) + len(additional_headers) + 1, f"Plate {plate_number}", header_format)
        row_index += 1
        for col_index, header in enumerate([''] + column_headers + additional_headers, start=1):
            worksheet.write(row_index, col_index, header, header_format)
        row_index += 1
        for row, row_data in sorted(plate_data['rows'].items()):
            worksheet.write(row_index, 1, row, header_format)
            for col_num, (well_id, well_data) in enumerate(row_data['wells'].items(), start=2):
                worksheet.write(row_index, col_num, well_data.get('abs_val', ""), cell_format)
            first_well = next(iter(row_data['wells'].values()))
            well_data = [first_well.get('synth_id', ''),
                        first_well.get('amc', ''),
                        first_well.get(assay_column_header.lower(), '')] + \
                        ([] if assay == "hc50" or assay == "cc50" else ['']) + \
                        [first_well.get('note', '')]
            for col_num, data in enumerate(well_data, start=len(column_headers) + 2):
                worksheet.write(row_index, col_num, data, cell_format)
            row_index += 1
        min_color = "#E599FF" if assay == 'cc50' else "#FFFFFF"
        max_color = "#F94449" if assay == "hc50" else ("#FFCCCC" if assay == "cc50" else "#FFE599")
        colour_scale_option = {
            'type': '2_color_scale',
            'min_color': min_color,
            'max_color': max_color}
        worksheet.conditional_format(row_index - len(plate_data['rows']),
                                     1, row_index - 1, len(column_headers) + 1,
                                     colour_scale_option)
        row_index += 2
    progress_bar.next()

def create_mic_c50_sheet_manual(master_dict, writer, assay, progress_bar):
    """Create the MIC-MBC, HC50, or CC50 Excel sheet for manual assay operation mode."""
    date_str = datetime.now().strftime('%Y%m%d')
    sheet_name_mapping = {'ast': f'MIC-MBC_{date_str}',
                          'hc50': f'HC50_{date_str}',
                          'cc50': f'CC50_{date_str}'}
    sheet_name = sheet_name_mapping[assay]
    workbook, worksheet, header_format, cell_format = initialize_worksheet(writer, sheet_name)
    headers = ['Synth_ID', 'AMC']
    if assay == 'ast':
        headers += ['MIC', 'MBC']
    else:
        headers.append(assay.upper())
    for col_num, header in enumerate(headers, start=0):
        worksheet.write(0, col_num, header, header_format)
    row_num = 1
    for plate_data in master_dict['plates'].values():
        for row_data in plate_data['rows'].values():
            first_well_key = next(iter(row_data['wells']))
            first_well_data = row_data['wells'][first_well_key]
            row_values = [first_well_data.get('synth_id', ''), first_well_data.get('amc', '')]
            if assay == 'ast':
                row_values += [first_well_data.get('mic', 'N/A'), first_well_data.get('mbc', '')]  # Placeholder for MBC
            else:
                row_values.append(first_well_data.get(assay, 'N/A'))
            for col_num, value in enumerate(row_values, start=0):
                worksheet.write(row_num, col_num, value, cell_format)
            row_num += 1
    worksheet.set_column('A:A', 14)  # Synth_ID
    worksheet.set_column('B:B', 9)   # AMC
    worksheet.set_column('C:D', 6)   # Concentration values
    progress_bar.next()

def create_visual_data_sheet_hts(master_dict, assay, writer, progress_bar):
    """ Create the "Visual_Data" sheet """
    workbook, worksheet, header_format, cell_format = initialize_worksheet(writer, 'Visual-Data')
    reference_plate_wells = next(iter(master_dict['plates'].values()))['wells'].keys()
    tech_reps = {plate_data['tech_rep'] for plate_data in master_dict['plates'].values()}
    first_row = 0
    first_col = 0
    for index, tech_rep in enumerate(tech_reps):
        plates_in_rep = sorted(
            [plate_data for plate_data in master_dict['plates'].values() if plate_data['tech_rep'] == tech_rep],
            key=lambda x: x['conc_val'],
            reverse=True)
        result_header = 'MIC' if assay == 'ast' else 'HC50'
        headers = ['AMC'] + [str(plate['conc_val']) for plate in plates_in_rep] + [result_header]
        for col, header in enumerate(headers, start=first_col):
            worksheet.write(first_row, col, header, header_format)
        worksheet.set_column(first_col, first_col, 12.5)
        header_row = first_row
        first_row += 1
        for well in reference_plate_wells:
            well_amc_data = [plates_in_rep[0]['wells'][well].get('amc', 'N/A')]
            well_amc_data += [plate['wells'][well]['abs_val'] for plate in plates_in_rep]
            assay_result_key = 'mic' if assay == 'ast' else 'hc50'
            well_amc_data.append(plates_in_rep[0]['wells'][well].get(assay_result_key, 'N/A'))
            for col, item in enumerate(well_amc_data, start=first_col):
                worksheet.write(first_row, col, item, cell_format)
            first_row += 1
        abs_first_col = first_col + 1
        abs_last_col = first_col + len(plates_in_rep)
        max_colour = "#F94449" if assay == "hc50" else "#FFE599"
        min_colour = "#FFFFFF"
        for col in range(abs_first_col, abs_last_col + 1):
            worksheet.conditional_format(header_row + 1, col, first_row - 1, col, {
                'type': '2_color_scale',
                'min_color': min_colour,
                'max_color': max_colour})
        # Adjust start column for the next technical replicate
        first_col += len(headers) + 2
        first_row = 0
    progress_bar.next()

def create_mic_hc50_sheet_hts(master_dict, assay, writer, progress_bar):
    """ Create the "MIC-HC50" sheet """
    current_date = datetime.now().strftime('%Y%m%d')
    sheet_name = 'HC50-' + current_date if assay == 'hc50' else 'MIC-MBC_' + current_date
    workbook, worksheet, header_format, cell_format = initialize_worksheet(writer, sheet_name)
    assay_col_prefix = 'HC50' if assay == 'hc50' else 'MIC'
    assay_result_key = assay_col_prefix.lower()
    headers = ['Well', 'AMC', 'Synthesis ID'] + [f"{assay_col_prefix}-R{i+1}" for i in range(args.num_tech_rep)]
    worksheet.set_column(0, len(headers) - 1, 15)
    if assay == 'ast':
        headers.append('MBC')
    headers += ['Notes']
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    notes_column_index = len(headers) - 1
    worksheet.set_column(notes_column_index, notes_column_index, 70)
    well_data_compiled = {}
    for plate_data in master_dict['plates'].values():
        for well, well_data in plate_data['wells'].items():
            if well_data.get('amc') in ["Growth Control", "Sterility Control"]:
                continue
            if well not in well_data_compiled:
                well_data_compiled[well] = {
                    'amc': well_data.get('amc', 'N/A'),
                    'synth_id': well_data.get('synth_id', 'N/A'),
                    'values': [None] * args.num_tech_rep,
                    'notes': well_data.get('note', '')}
            if well_data_compiled[well]['values'][plate_data['tech_rep'] - 1] is None:
                well_data_compiled[well]['values'][plate_data['tech_rep'] - 1] = well_data.get(assay_result_key, 'N/A')
    row = 1
    for well, data in well_data_compiled.items():
        values = [v if v is not None else 'N/A' for v in data['values']]
        worksheet.write(row, 0, well, cell_format)
        worksheet.write(row, 1, data['amc'], cell_format)
        worksheet.write(row, 2, data['synth_id'], cell_format)
        for i, value in enumerate(values):
            worksheet.write(row, 3 + i, value, cell_format)
        if assay == 'ast':
            worksheet.write(row, 3 + args.num_tech_rep,'', cell_format)
        worksheet.write(row, 4 + args.num_tech_rep if assay == 'ast' else 3 + args.num_tech_rep, data['notes'], cell_format)
        row += 1
    progress_bar.next()

def create_raw_plate_sheet(master_dict, assay, writer,progress_bar):
    """ Create the "Raw-Data" sheet """
    workbook, worksheet, header_format, cell_format = initialize_worksheet(writer, 'Raw-Data')
    row_letters = sorted(set(well[0] for plate in master_dict['plates'].values() for well in plate['wells']))
    column_numbers = sorted(set(int(well[1:]) for plate in master_dict['plates'].values() for well in plate['wells']))
    worksheet.set_column('A:M', 6)
    first_row = 0
    for plate_number, plate_data in master_dict['plates'].items():
        worksheet.merge_range(first_row, 1, first_row, len(column_numbers), f"Plate {plate_number} - Replicate {plate_data['tech_rep']}", header_format)
        first_row += 1
        for col_num, col_header in enumerate(column_numbers, start=1):
            worksheet.write(first_row, col_num, col_header, header_format)
        for row_num, row_letter in enumerate(row_letters, start=first_row + 1):
            worksheet.write(row_num, 0, row_letter, header_format)
            for col_num, col_header in enumerate(column_numbers, start=1):
                well = f"{row_letter}{col_header}"
                value = plate_data['wells'][well]['abs_val'] if well in plate_data['wells'] else ""
                worksheet.write(row_num, col_num, value, cell_format)
        worksheet.conditional_format(first_row + 1, 1, row_num, len(column_numbers), {
            'type': '2_color_scale',
            'min_color': "#FFFFFF",
            'max_color': "#F94449" if assay == "hc50" else "#FFE599"})
        first_row = row_num + 2
    progress_bar.next()

def main():
    """
    main
    """
    master_dict, num_plates = process_data(args.raw_data, args.matrix, args.operation_mode)
    plates_per_rep = num_plates // args.num_tech_rep

    if args.operation_mode == 'manual':
        assign_manual_rep(master_dict)
        conc_list = calculate_concentrations(args.start_con, args.final_con, plates_per_rep, args.operation_mode, master_dict)
        assign_conc_manual(master_dict, conc_list)

    # Calculate concetration
    if args.operation_mode == 'hts':
        conc_list = calculate_concentrations(args.start_con, args.final_con, plates_per_rep, args.operation_mode, master_dict)
        populate_rep_conc_hts(master_dict, plates_per_rep, conc_list)

    # Validate the inputs
    if not validate_input(args.num_tech_rep, args.start_con, args.final_con, num_plates, conc_list, args.operation_mode):
        return

    # Compute assay
    compute_selected_assay(args.assay, master_dict, args.operation_mode)

    # Generate Excel file
    generate_excel(master_dict, args.assay, args.prefix, args.operation_mode, args.output_dir)

if __name__ == "__main__":
    main()
