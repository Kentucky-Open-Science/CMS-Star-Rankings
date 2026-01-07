
import csv
import calculations

def generate_template():
    # flattened list of all measures
    measures = []
    for group in calculations.MEASURE_GROUPS.values():
        measures.extend(group)
    
    headers = ['Provider ID', 'Hospital Name'] + sorted(measures)
    
    with open('hospital_data_template.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        # Add a sample row
        sample_row = ['123456', 'Example Hospital'] + [None] * len(measures)
        writer.writerow(sample_row)

    print(f"Template generated with {len(headers)} columns.")

if __name__ == "__main__":
    generate_template()
