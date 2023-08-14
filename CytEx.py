import pandas as pd
import os
import numpy as np
from datetime import date


# Ask the user to select the assay
print("Select the assay:")
print("1. hAST")
print("2. mAST")

assay = input()

### hAST
if assay == "1":
    # Ask for the organism name
    organism_name = input("Enter the organism name used in the experiment: ")

    # Replace spaces in the organism name with underscores
    organism_name = organism_name.replace(" ", "_")

    # Create unique folder name with the current date
    folder_name = f"{organism_name}_{date.today().strftime('%m_%d_%Y')}"

    # Ask the user for the raw file paths and validate them
        # Function to validate whether the given file path is an Excel file
    def is_excel_file(file_path):
        return file_path.endswith('.xlsx') or file_path.endswith('.xls')
    
    while True:
        file_path = input('Please enter the path of raw plate read data: ')
        if not is_excel_file(file_path) or not os.path.exists(file_path):
            print('Invalid file path or the file is not in Excel format. Please try again.')
        else:
            break

    while True:
        matrix_map_path = input('Please enter the file path of the matrix map: ')
        if not is_excel_file(matrix_map_path) or not os.path.exists(matrix_map_path):
            print('Invalid file path or the file is not in Excel format. Please try again.')
        else:
            break

    raw_data = pd.read_excel(file_path)
    matrix_map = pd.read_excel(matrix_map_path)

    # Convert dataframes to numpy arrays
    raw_data_array = raw_data.to_numpy()

    # Initialize plate count and plate_data_raw dictionary
    plates = 0
    plate_data_raw = {}

    # Get the dimensions of the raw data array
    rows, cols = raw_data_array.shape

    # Find the starting point of each plate in the raw data array and extract the plate data
    
    for row in range(rows - 1):
        for col in range(cols - 1):
            cell = raw_data_array[row, col]
            cell_south = raw_data_array[row + 1, col]
            cell_east = raw_data_array[row, col + 1]

            if pd.isna(cell) and cell_south == "A" and cell_east == 1:
                plates += 1
                raw_plate_name = f"Plate {plates}"
                plate_data_raw[raw_plate_name] = {}

                for i in range(8):
                    for j in range(12):
                        cell_name = chr(ord('A') + i) + str(j + 1)
                        plate_data_raw[raw_plate_name][cell_name] = raw_data_array[row + 1 + i, col + 1 + j]

    # Print the total number of plates retrieved
    print(f"Total number of plates retrieved from the excel file: {plates}")


    # Function to request user input and validate it
    def request_input():
        while True:
            technical_replicates = int(input("Enter the number of technical replicates: "))
            starting_concentration = float(input("Enter the starting concentration (µg/ml): "))
            final_concentration = float(input("Enter the final concentration (µg/ml): "))

            plates_per_technical_replicate = plates // technical_replicates

            # Check if the starting concentration is less than or equal to the final concentration
            if starting_concentration <= final_concentration:
                print("Starting concentration should be greater than final concentration. Please try again.")
                continue

            # Check if the number of technical replicates exceeds the number of plates
            if technical_replicates > plates:
                print("Number of technical replicates exceeds the number of plates. Please try again.")
                continue

            # Check if the dilution series corresponds to a 1:2 dilution for each step given the starting and final concentrations and the number of plates per group
            dilution_steps = np.log2(starting_concentration / final_concentration) + 1
            if not np.isclose(dilution_steps, plates_per_technical_replicate):
                print(
                    "The dilution series does not correspond to a 1:2 dilution for each step given the starting and final concentrations and the number of plates per technical replicate. Please try again.")
                continue

            # Check if the number of plates is not divisible by the number of technical replicates (i.e., there are leftover plates)
            if plates % technical_replicates != 0:
                print(
                    "The number of plates is not evenly divisible by the number of provided technical replicates. This will result in some plates being unused. Please try again.")
                continue

            return technical_replicates, starting_concentration, final_concentration, plates_per_technical_replicate


    # Obtain validated user inputs
    technical_replicates, starting_concentration, final_concentration, plates_per_technical_replicate = request_input()

    # Calculate the concentrations for each plate
    concentrations = np.logspace(np.log10(starting_concentration), np.log10(final_concentration), plates_per_technical_replicate)
    concentrations = [round(c, 1) for c in concentrations]

    # Rename plate data dictionary keys and update the values with new information
    plate_data = {}

    for replicate in range(1, technical_replicates + 1):
        for plate in range(1, plates_per_technical_replicate + 1):
            old_plate_name = f"Plate {(replicate - 1) * plates_per_technical_replicate + plate}"
            plate_name = f"{replicate}.{plate}"
            plate_data[plate_name] = plate_data_raw[old_plate_name]

            for well, value in plate_data[plate_name].items():
                # Extract the Synthesis_ID and BIROL_ID for the current well from the matrix_map
                synthesis_id = matrix_map[matrix_map['Well'] == well]['Synthesis_ID'].values[0]
                birol_id = matrix_map[matrix_map['Well'] == well]['BIROL_ID'].values[0]

                # Add additional data to the plate_data dictionary for the current well
                plate_data[plate_name][well] = {
                    "value": value,
                    "technical_replicate": replicate,
                    "concentration": concentrations[plate - 1],
                    "Synthesis_ID": synthesis_id,
                    "BIROL_ID": birol_id
                }

    # Define output paths
    output_folder = os.path.join(os.getcwd(), "output")

    # Create output directory for the current run
    output_folder = os.path.join(output_folder, folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define sub-directories for "plates" and "replicates"
    plates_folder = os.path.join(output_folder, "plates")
    replicates_folder = os.path.join(output_folder, "replicates")

    # Create sub-directories if they don't exist
    if not os.path.exists(plates_folder):
        os.makedirs(plates_folder)
    if not os.path.exists(replicates_folder):
        os.makedirs(replicates_folder)

    # Write each plate's data to a separate CSV file in the "plates" directory
    for plate_name, cell_data in plate_data.items():
        # Create a DataFrame for the current plate's data
        plate_df = pd.DataFrame.from_dict(cell_data, orient='index')
        plate_df.index.name = 'Well'

        # Generate the output file path for the current plate
        output_file_csv = os.path.join(plates_folder, f"{plate_name}.csv")

        # Save the plate's data to a CSV file
        plate_df.to_csv(output_file_csv)

    # Iterate over each technical replicate
    for replicate in range(1, technical_replicates + 1):
        # Initialize a DataFrame to store the results for this replicate
        replicate_df = pd.DataFrame()

        # Extract the wells from the plate data
        wells = list(plate_data[f"{replicate}.1"].keys())

        # Initialize each row with well name
        replicate_df['Well'] = wells

        # Add the "AMP" column and populate it with corresponding BIROL_ID
        replicate_df['AMP'] = [plate_data[f"{replicate}.1"][well]['BIROL_ID'] for well in wells]

        # Iterate over each dilution stage
        for dilution_stage in range(1, plates_per_technical_replicate + 1):
            # Get the concentration for the current dilution stage
            concentration = concentrations[dilution_stage - 1]

            # Prepare the column name with concentration
            column_name = str(concentration)

            # Extract the values for each well at the current dilution stage
            plate_name = f"{replicate}.{dilution_stage}"
            dilution_values = [plate_data[plate_name][well]['value'] for well in wells]

            # Add the new column to our result DataFrame
            replicate_df[column_name] = dilution_values


        # Add a new column for MIC and update plate_data dictionary
        def calculate_mic(row):
            if row['AMP'] in ['Sterility Control', 'Growth Control']:
                return 'N/A'
            else:
                return next((c for c, v in reversed(list(zip(concentrations, row[2:]))) if round(v, 2) < 0.4999),
                            f">{int(starting_concentration)}")


        replicate_df['MIC'] = replicate_df.apply(calculate_mic, axis=1)

        for index, row in replicate_df.iterrows():
            well = row['Well']
            mic = row['MIC']
            for plate_name, well_data in plate_data.items():
                if well in well_data:
                    plate_data[plate_name][well]['MIC'] = mic

        # Generate the output file paths for the current replicate
        replicate_df_csv = os.path.join(replicates_folder, f"Replicate_{replicate}.csv")
        replicate_df_excel = os.path.join(replicates_folder, f"Replicate_{replicate}.xlsx")

        # Write replicate's data to CSV and Excel files
        replicate_df.to_csv(replicate_df_csv, index=False)
        replicate_df.to_excel(replicate_df_excel, index=False)


### Combined MIC
    # Initialize an empty DataFrame to store combined data
    combined_df = pd.DataFrame()

    # Separate the data by technical replicate group
    for group in range(1, technical_replicates + 1):
        # Specify the input CSV file
        group_csv_input = os.path.join(output_folder, "replicates", f"Replicate_{group}.csv")

        # Read the data from the CSV file
        group_df = pd.read_csv(group_csv_input)

        # Remove the rows where BIROL_ID is "Sterility Control" or "Growth Control"
        group_df = group_df.loc[~group_df['AMP'].isin(["Sterility Control", "Growth Control"])]

        # Add a new column for the technical replicate group number
        group_df['Technical_Replicate_Group'] = group

        # Reorder the columns to have 'Well', 'AMP', 'Technical_Replicate_Group', and 'MIC'
        group_df = group_df[['Well', 'AMP', 'Technical_Replicate_Group', 'MIC']]

        # Append the group_df to the combined_df
        combined_df = pd.concat([combined_df, group_df], ignore_index=True)

    # Define sub-directory for "combined"
    combined_folder = os.path.join(output_folder, "combined")

    # Create sub-directory if it doesn't exist
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Specify the output CSV file for combined data
    combined_csv_output = os.path.join(combined_folder, "Combined_MIC_Values.csv")

    # Write the combined DataFrame to a CSV file
    combined_df.to_csv(combined_csv_output, index=False)

    # Specify the output Excel file for combined data
    combined_excel_output = os.path.join(combined_folder, "Combined_MIC_Values.xlsx")

    # Write the combined DataFrame to an Excel file
    combined_df.to_excel(combined_excel_output, index=False)

### MIC Plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Output dimensions
    plot_width = 20
    plot_height = 8
    dpi = 800

    # Make MIC non-numeric for plotting
    combined_df['MIC'] = combined_df['MIC'].astype(str)

    # Create concentration list with string type
    concentration_list = [f">{int(starting_concentration)}"] + [str(i) for i in concentrations]

    # Order y-axis
    combined_df['MIC'] = pd.Categorical(combined_df['MIC'], ordered=True, categories=concentration_list)

    # Initialize the figure and axes
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # Scatter plot
    scatter_plot = sns.stripplot(x='AMP', y='MIC', hue='Technical_Replicate_Group', palette='Set1', data=combined_df,
                                ax=ax, s=10, jitter=True, dodge=True)

    ax.set_xlabel('AMP', fontsize=15, fontweight='bold')
    ax.set_ylabel('MIC (µg/mL)', fontsize=15, fontweight='bold')

    ax.legend(title='Technical Replicate', title_fontsize='13', loc='upper right')

    ax.tick_params(axis='x', labelrotation=90, labelsize=8)  # Rotate x-axis labels and decrease font size

    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10, title="Technical Replicate")
    for handle in lgnd.legend_handles:
        handle.set_sizes([30.0])

    plt.tight_layout()

        # Define sub-directory for "plot"
    plot_folder = os.path.join(output_folder, "plot")

    # Create sub-directory if it doesn't exist
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Save MIC plot
    mic_plot_path = os.path.join(plot_folder, "MIC_plot.png")
    plt.savefig(mic_plot_path, format='png', dpi=300)

### Excel Output
    import datetime
    import os
    import pandas as pd
    import seaborn as sns

    # Get the current date
    now = datetime.datetime.now()

    # Ensure the "output" directory exists within the output_folder, if not, create it
    final_output_folder = os.path.join(output_folder, "output")
    os.makedirs(final_output_folder, exist_ok=True)

    # Create an Excel writer object with the path to the "output" directory
    output_filename = os.path.join(final_output_folder, f"{organism_name}_MICMBC_{now.strftime('%m-%d-%Y')}.xlsx")

    # Specify the number of empty columns to leave between replicates
    empty_columns = 1

    # The column offset for each replicate
    offset = 0

    # Generate a color palette
    colors = sns.color_palette("Set2", n_colors=technical_replicates)

    # Open Excel writer and Visual_Results worksheet within 'with' block
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Define a format for the cell borders
        border_format = writer.book.add_format({'border': 1, 'align': 'center'})

        # Define a format for the headers in the MICMBC tab
        header_format = writer.book.add_format(
            {'bold': True, 'bg_color': '#D3D3D3', 'align': 'center', 'bottom': 2, 'bottom_color': 'black'})

        # Add a worksheet for Visual_Results
        worksheet = writer.book.add_worksheet('Visual_Results')

        first_replicate_well_amp = None

        # Iterate over each technical replicate
        for replicate in range(1, technical_replicates + 1):
            # Read the replicate's data from the Excel file
            replicate_df = pd.read_excel(os.path.join(output_folder, "replicates", f"Replicate_{replicate}.xlsx"))

            # Fill empty cells with 'N/A'
            replicate_df.fillna('N/A', inplace=True)

            # Save Well-AMP pairs from the first replicate
            if replicate == 1:
                first_replicate_well_amp = replicate_df[['Well', 'AMP']].values.tolist()

            # Write the DataFrame to the "Visual_Results" worksheet
            replicate_df.to_excel(writer, sheet_name='Visual_Results', startcol=offset, index=False)

            # Get the last row and last column index for formatting
            last_row = len(replicate_df.index)
            last_col = len(replicate_df.columns)

            # Get the xlsxwriter workbook and worksheet objects.
            workbook = writer.book
            worksheet = writer.sheets['Visual_Results']

            # Assign the color for this replicate
            color = colors[replicate - 1]
            hex_color = '#%02x%02x%02x' % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            # Apply the conditional formatting rule to the required range except the 'MIC' column
            for col_num in range(offset, offset + last_col):
                if replicate_df.columns[col_num - offset] != 'MIC':
                    worksheet.conditional_format(1, col_num, last_row, col_num, {'type': '2_color_scale',
                                                                                 'min_value': '0',
                                                                                 'max_value': '100',
                                                                                 'min_color': "#FFFFFF",
                                                                                 'max_color': hex_color})

                # Apply the border format to all cells including 'MIC'
                for row_num in range(1, last_row + 1):  # +1 to include the last row
                    cell_value = replicate_df.iloc[row_num - 1, col_num - offset]
                    worksheet.write(row_num, col_num, cell_value, border_format)

            # Increase the offset for the next replicate (the width of the DataFrame plus the number of empty columns)
            offset += len(replicate_df.columns) + empty_columns


        # Add a worksheet for MICMBC with the current date in the name
        micmbc_worksheet = writer.book.add_worksheet(f"MICMBC_{now.strftime('%m%d%Y')}")

        # Set the format for all cells to be centered
        centered_format = writer.book.add_format({'align': 'center'})

        # Write the headers for MICMBC with centered alignment
        micmbc_worksheet.write(0, 0, 'Well', header_format)
        micmbc_worksheet.write(0, 1, 'AMP', header_format)

        # Write the MIC data for each technical replicate with centered alignment
        for replicate in range(1, technical_replicates + 1):
            # Read the replicate's data from the Excel file
            replicate_df = pd.read_excel(os.path.join(output_folder, "replicates", f"Replicate_{replicate}.xlsx"))

            # Fill empty cells with 'N/A'
            replicate_df.fillna('N/A', inplace=True)

            # Filter out 'N/A' values in the MIC column for the "MICMBC" tab
            replicate_df = replicate_df[replicate_df['MIC'] != 'N/A']

            # Get the MIC column name for the current replicate
            mic_column_name = f"MIC_R{replicate}"

            # Write the MIC values to the "MICMBC" worksheet with centered alignment
            micmbc_worksheet.write(0, replicate + 1, mic_column_name, header_format)
            for i, mic_value in enumerate(replicate_df['MIC'], start=1):
                micmbc_worksheet.write(i, 0, replicate_df.iloc[i - 1]['Well'], centered_format)
                micmbc_worksheet.write(i, 1, replicate_df.iloc[i - 1]['AMP'], centered_format)
                micmbc_worksheet.write(i, replicate + 1, mic_value, centered_format)

        # Write the MBC column after the last MIC data (cells below left empty)
        micmbc_worksheet.write(0, technical_replicates + 2, 'MBC', header_format)

        # Prepare a dictionary to store lowest MIC and corresponding replicate for each well
        well_mic_data = {row[0]: {'L_MIC': float('inf'), 'R': None} for row in first_replicate_well_amp}

        # Add a worksheet for MBC_Wells
        mbc_wells_worksheet = writer.book.add_worksheet('MBC_Wells')

        # Define header format
        header_format = writer.book.add_format({
            'bold': True,
            'align': 'center',
            'bg_color': '#D3D3D3',  # light gray fill
            'bottom': 2,  # thick bottom border
        })

        # Write the column headers to the MBC_Wells worksheet
        mbc_wells_worksheet.write('A1', 'Well', header_format)
        mbc_wells_worksheet.write('B1', 'L_MIC', header_format)
        mbc_wells_worksheet.write('C1', 'R', header_format)

        # Calculate lowest MIC and its replicate for each well
        for replicate in range(1, technical_replicates + 1):
            # Read the replicate's data from the Excel file
            replicate_df = pd.read_excel(os.path.join(output_folder, "replicates", f"Replicate_{replicate}.xlsx"))

            # Fill empty cells with 'N/A'
            replicate_df.fillna('N/A', inplace=True)

            # Iterate over all wells
            for _, row in replicate_df.iterrows():
                well = row['Well']
                mic = str(row['MIC'])  # Convert MIC to string

                # If MIC value is numeric and less than current lowest MIC for the well, update the lowest MIC and corresponding replicate
                if mic.replace('.', '', 1).isdigit() and float(mic) < well_mic_data[well]['L_MIC']:
                    well_mic_data[well]['L_MIC'] = float(mic)
                    well_mic_data[well]['R'] = replicate

        # Write the data for each well in the MBC_Wells worksheet
        i = 1  # Start writing from the second row to leave the first row for headers
        for well, mic_data in well_mic_data.items():
            # If lowest MIC for the well is still inf (i.e., all replicates contain > in their mic values), skip this well
            if mic_data['L_MIC'] == float('inf'):
                continue

            # Write the Well, L_MIC, and R values to the MBC_Wells worksheet with centered alignment
            mbc_wells_worksheet.write(i, 0, well, centered_format)
            mbc_wells_worksheet.write(i, 1, mic_data['L_MIC'], centered_format)
            mbc_wells_worksheet.write(i, 2, mic_data['R'], centered_format)
            i += 1  # Increment the row index

        # Add a worksheet for Plots
        plot_worksheet = writer.book.add_worksheet('Plots')

        # Set the width of the columns in the Plots worksheet
        plot_worksheet.set_column(0, 0, 30)
        plot_worksheet.set_column(1, 1, 30)

        # Insert the MIC plot
        mic_plot_path = os.path.join(plot_folder, "MIC_plot.png")
        plot_worksheet.insert_image('A1', mic_plot_path, {'x_scale': 1, 'y_scale': 1})


    print(f'Output written to {output_filename}')
        
### mAST
elif assay == "2":
    print("mAST data extraction is not currently avaliable!")
    pass
