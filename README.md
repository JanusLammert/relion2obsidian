# RELION2Obsidian

A Python tool to convert RELION cryo-EM project data into an Obsidian vault for improved organization, documentation, and visualization.

## Overview

RELION2Obsidian creates a navigable knowledge base from your RELION project data by:
- Extracting metadata from all RELION job types
- Generating interconnected Markdown notes for Obsidian
- Creating visual representations of 2D class averages
- Building a network of relationships between processing steps
- Providing an indexed, searchable project history

## Features

- **Automatic Job Detection**: Identifies all RELION job types in your project directory
- **Relationship Mapping**: Creates bidirectional links between dependent jobs
- **Visual Content**:
  - Generates montages of 2D class averages
  - Preserves job parameters and command-line settings
- **Knowledge Organization**:
  - Tags jobs by type and function
  - Creates an index page for easy navigation
  - Preserves user notes from the original project

## Installation

### Prerequisites

- Python 3.7+
- An [Obsidian](https://obsidian.md/) installation (for viewing the generated vault)

### Required Python Packages

```bash
pip install numpy matplotlib mrcfile starfile scikit-image pillow tqdm
```

## Usage

### Basic Usage

```bash
python rel2obsi.py -i /path/to/relion/project -o /path/to/output/obsidian/vault
```

### Command-Line Options

```
usage: rel2obsi.py [-h] -i PROJECT_DIR -o OUTPUT_DIR [--force] [--verbose]

Generate Obsidian notes from RELION project jobs.

optional arguments:
  -h, --help            show this help message and exit
  -i PROJECT_DIR, --project_dir PROJECT_DIR
                        Path to the RELION project directory.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to the output directory for Obsidian notes.
  --force               Force regeneration of existing notes.
  --verbose             Enable verbose logging.
```

### Example

```bash
python rel2obsi.py -i ~/cryo_em/ribosome_project -o ~/Documents/obsidian/ribosome_vault --verbose
```

## Viewing the Results

1. Open Obsidian
2. Select "Open folder as vault"
3. Navigate to your output directory
4. Start exploring from the `00_RELION_Project_Index.md` file

## Understanding the Generated Vault

### Note Structure

Each RELION job gets its own note with the following sections:

- **Job Information**: Basic metadata about the job
- **Input Jobs**: Links to upstream processing steps
- **Jobs that use this**: Links to downstream processing steps
- **User Notes**: Any notes you added in RELION
- **Additional Details**: Extended metadata
- **Highlighted Settings**: Important parameters at a glance
- **All Settings**: Complete command-line parameters
- **Job-specific sections**: Visual content for certain job types

### Job Type-Specific Features

- **Class2D**: Montage images of 2D classes 
- **Class3D**: Placeholder for 3D classification images
- **Refine3D**: Placeholder for refinement statistics and automatic linking to PostProcess jobs

## Troubleshooting

Common issues and solutions:

- **Missing dependencies**: Make sure all required Python packages are installed
- **File permissions**: Ensure write access to the output directory
- **Memory issues**: For very large projects, try processing subsets of the data

If you encounter errors, check the `relion_to_obsidian.log` file for detailed information.

## Advanced Usage

### Integrating with Existing Workflows

You can run this tool periodically to keep your documentation in sync with your ongoing analysis:

```bash
python rel2obsi.py -i /path/to/relion/project -o /path/to/existing/obsidian/vault
```

The tool will update only what has changed since the last run.

### Force Refresh All Notes

To regenerate all notes (useful after code updates):

```bash
python rel2obsi.py -i /path/to/relion/project -o /path/to/obsidian/vault --force
```

## Development Status

This tool is currently in active development. Version 9 is the latest but has limited testing. User feedback and bug reports are welcome!

## License

[MIT License](LICENSE)

## Citation

If you use this tool in your research, please cite:
```
RELION2Obsidian: A tool for converting RELION cryo-EM projects to Obsidian knowledge bases
```
