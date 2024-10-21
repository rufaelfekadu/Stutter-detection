import yaml
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class_abbreviations = {
    'Block': 'BL',
    'FacialGrimaces': 'FG',
    'HeadMovement': 'HM',
    'IncompeleteSyllableRepetition': 'ISR',
    'MovementOfExtremities': 'ME',
    'MultisyllabicUnitRepetition': 'MUR',
    'Prolongation': 'PR',
    'SoundRepetition': 'SR',
    'Verbal': 'VB'
}

def create_notion_table(metrics_data: dict, metric_type: str) -> str:
    """Create a Notion-compatible table with exact formatting."""
    all_classes = sorted(set().union(*[set(m.keys()) for m in metrics_data.values()]))
    
    # Create header row with exact spacing
    header = "| File | " + " | ".join(class_abbreviations.get(c, c[:2].upper()) for c in all_classes) + " |"
    
    # Create separator with exact format
    separator = "|----|" + "|".join("-" * len(abbrev) for abbrev in 
                                  [class_abbreviations.get(c, c[:2].upper()) for c in all_classes]) + "|"
    
    # Build table rows
    rows = []
    for filename, metrics in metrics_data.items():
        row_values = [filename]
        for class_name in all_classes:
            if class_name in metrics:
                class_data = metrics[class_name]
                if metric_type == 'f_measure':
                    value = f"{class_data['f_measure']['f_measure']*100:.1f}"
                else:  # Nref or Nsys
                    value = str(int(class_data['count'][metric_type]))
                row_values.append(value)
            else:
                row_values.append("-")
        rows.append("| " + " | ".join(row_values) + " |")
    
    # Combine all parts
    return "\n".join([header, separator] + rows)

def create_table(metrics_data: Dict, metric_type: str) -> str:
    """Create a table for a specific metric type (F-measure, Nref, or Nsys)."""
    # Get all unique classes
    all_classes = sorted(set().union(*[set(m.keys()) for m in metrics_data.values()]))
    
    # Create header
    header = "| File |"
    separator = "|------|"
    
    # Add abbreviated class names to header
    for class_name in all_classes:
        abbrev = class_abbreviations.get(class_name, class_name[:2].upper())
        header += f" {abbrev} |"
        separator += "------|"
    
    table = f"{header}\n{separator}\n"
    
    # Add rows
    for filename, metrics in metrics_data.items():
        row = [filename]
        for class_name in all_classes:
            if class_name in metrics:
                class_data = metrics[class_name]
                if metric_type == 'f_measure':
                    value = f"{class_data['f_measure']['f_measure']*100:.1f}%"
                else:  # Nref or Nsys
                    value = str(int(class_data['count'][metric_type]))
                row.append(value)
            else:
                row.append("-")
        table += "| " + " | ".join(row) + " |\n"
    
    return table

def process_multiple_yaml_files(yaml_files: List[str]) -> Tuple[str, str, str]:
    """Process multiple YAML files and create three separate tables."""
    event_based_metrics = {}
    segment_based_metrics = {}
    # Process each YAML file
    for file_path in yaml_files:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            filename = Path(file_path).stem
            class_metrics = data['event_based_metrics']['class_wise']
            event_based_metrics[filename] = class_metrics
            segment_based_metrics[filename] = data['segment_based_metrics']['class_wise']
    # Create three separate tables
    event_based = "# Event-Based Table\n" + create_notion_table(event_based_metrics, 'f_measure')
    segment_based = "\n# Segment-Based Table\n" + create_notion_table(segment_based_metrics, 'f_measure')

    # nref_table = "\n# Nref Table\n" + create_notion_table(all_metrics, 'Nref')
    # nsys_table = "\n# Nsys Table\n" + create_notion_table(all_metrics, 'Nsys')
    
    return event_based, segment_based

def main(args):
    # Find all YAML files in current directory
    output_dir = Path("outputs/fluencybank/annotator_evaluations/reading")
    yaml_files = glob.glob(f"{args.yaml_files}/*.yaml")
    
    if not yaml_files:
        print("No YAML files found in current directory!")
        return
    
    print("Processing YAML files:", yaml_files)
    event_based_table, segment_based_table = process_multiple_yaml_files(yaml_files)
    
    # Save all tables to file
    with open(f"{args.yaml_files}/tables.md", "w") as f:
        f.write(event_based_table)
        f.write(segment_based_table)
    
    print("\nTables preview:")
    print(event_based_table)
    print("\n" + "="*50 + "\n")
    print(segment_based_table)
    print("\nTables have been saved to 'tables.md'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tables from multiple YAML files')
    parser.add_argument('--yaml_files', type=str, default='outputs/fluencybank/annotator_evaluations/reading', help='Path to the YAML files')
    args = parser.parse_args()
    main(args)