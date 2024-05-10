import csv

# Define input and output file paths
input_file = '/home/grt/GloFE/OpenASL/openasl-v1.0.tsv'
output_file = '/home/grt/GloFE/OpenASL/openasl-v2.0.tsv'

# Open input and output files
with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    # Create CSV reader and writer objects
    reader = csv.DictReader(infile, delimiter='\t')
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter='\t')
    
    # Write header to output file
    writer.writeheader()
    
    # Iterate through rows in input file
    for row in reader:
        # Check if 'split' column value is 'test' or 'valid'
        if row['split'] in ['test', 'valid']:
            # Write row to output file
            writer.writerow(row)
