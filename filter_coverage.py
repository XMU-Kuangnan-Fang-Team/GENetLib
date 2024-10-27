import xml.etree.ElementTree as ET

def filter_coverage(input_file, output_file, functions):
    tree = ET.parse(input_file)
    root = tree.getroot()

    filtered_files = []
    for file in root.findall('.//file'):
        file_path = file.get('path')
        if any(func in file_path for func in functions):
            filtered_files.append(file)

    new_root = ET.Element("coverage")
    for file in filtered_files:
        new_root.append(file)

    tree = ET.ElementTree(new_root)
    tree.write(output_file)

if __name__ == "__main__":
    input_file = 'coverage.xml'
    output_file = 'coverage_filtered.xml'
    functions = ['sim_data_scalar', 'sim_data_func', 'func_ge', 'grid_func_ge']
    filter_coverage(input_file, output_file, functions)
