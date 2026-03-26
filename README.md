# RELION2Obsidian

A Python tool to convert RELION cryo-EM project data into an Obsidian vault for improved organization, documentation, and visualization.

## Overview

RELION2Obsidian creates a navigable knowledge base from your RELION project data by:
- Extracting metadata from all RELION job types
- Generating interconnected Markdown notes for Obsidian
- Creating visual representations of 2D/3D class averages and refinement statistics
- Building a network of relationships between processing steps
- Generating an interactive **Obsidian Canvas** with a tree-view of all jobs and their dependencies
- Providing an indexed, searchable project history

---

## Features

- **Automatic Job Detection**: Identifies all RELION job types in your project directory
- **Relationship Mapping**: Creates bidirectional links between dependent jobs, parsed directly from `default_pipeline.star`
- **Visual Content**:
  - Montages of 2D class averages with particle counts and resolution estimates
  - 3D class and refinement visualizations (map slices + orientation histograms)
  - CTF and motion correction logfile PDFs
- **Obsidian Canvas**:
  - Automatically generated job graph showing the full processing pipeline
  - Tree layout growing left-to-right, with parent jobs centered over their children
  - Color-coded nodes by job type; grey nodes indicate incomplete jobs
  - Configurable name and placement within your vault folder hierarchy
- **Knowledge Organization**:
  - Tags jobs by type and function
  - Creates an index page for easy navigation
  - Preserves user notes and command-line parameters from the original project
  - Tracks incomplete jobs across runs and updates notes when they complete

---

## Installation

### Prerequisites

- Python 3.11+
- An [Obsidian](https://obsidian.md/) installation (for viewing the generated vault)
- Both `rel2obsi_beta.py` and `canvas_generator.py` must be in the same directory

### Required Python Packages

Install all dependencies from **conda-forge** (recommended):

```bash
conda install -c conda-forge numpy matplotlib mrcfile starfile scikit-image pillow tqdm natsort
```

or using **Mamba** (faster):

```bash
mamba install -c conda-forge numpy matplotlib mrcfile starfile scikit-image pillow tqdm natsort
```

### Create a Dedicated Environment (Recommended)

```bash
conda create -n rel2obsi python=3.11 -c conda-forge numpy matplotlib mrcfile starfile scikit-image pillow tqdm natsort
conda activate rel2obsi
```

### Environment Setup via `environment.yml`

```bash
conda env create -f environment.yml
conda activate rel2obsi
```

Example `environment.yml`:

```yaml
name: rel2obsi
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - matplotlib
  - mrcfile
  - starfile
  - scikit-image
  - pillow
  - tqdm
  - natsort
```

---

## Vault Folder Structure

RELION2Obsidian writes Markdown notes into a subfolder of your Obsidian vault. The Canvas file (job graph) is saved separately, a configurable number of levels **above** the notes folder. This allows you to keep multiple projects or subprojects organized under one vault while all canvases remain at the top level where Obsidian can display them.

The recommended structure depends on how many organizational layers your work has.

### Single project (`--canvas-depth 1`)

```
vault/
├── MyProject.canvas          ← job graph canvas
└── MyProject/
    ├── 00_RELION_Project_Index.md
    ├── assets/
    │   └── *.png / *.pdf
    ├── job001_Import.md
    ├── job002_MotionCorr.md
    └── ...
```

Command:
```bash
python rel2obsi_beta.py \
  -i /data/relion/MyProject \
  -o /vault/MyProject \
  --canvas-name "MyProject" \
  --canvas-depth 1
```

### Project with subprojects (`--canvas-depth 2`, default)

This is the default setup. Multiple subprojects live under a shared project folder, and all canvases are placed at the vault root.

```
vault/
├── MyProject_Subproject_A.canvas     ← canvas for subproject A
├── MyProject_Subproject_B.canvas     ← canvas for subproject B
└── MyProject/
    ├── Subproject_A/
    │   ├── 00_RELION_Project_Index.md
    │   ├── assets/
    │   ├── job001_Import.md
    │   └── ...
    └── Subproject_B/
        ├── 00_RELION_Project_Index.md
        ├── assets/
        ├── job001_Import.md
        └── ...
```

Commands:
```bash
python rel2obsi_beta.py \
  -i /data/relion/MyProject/Subproject_A \
  -o /vault/MyProject/Subproject_A \
  --canvas-name "MyProject_Subproject_A" \
  --canvas-depth 2

python rel2obsi_beta.py \
  -i /data/relion/MyProject/Subproject_B \
  -o /vault/MyProject/Subproject_B \
  --canvas-name "MyProject_Subproject_B" \
  --canvas-depth 2
```

### Three-level hierarchy (`--canvas-depth 3`)

For very large projects with an additional grouping layer (e.g. dataset → condition → replicate).

```
vault/
├── DatasetX_ConditionA_Rep1.canvas
└── DatasetX/
    └── ConditionA/
        └── Rep1/
            ├── 00_RELION_Project_Index.md
            ├── assets/
            └── ...
```

Command:
```bash
python rel2obsi_beta.py \
  -i /data/relion/DatasetX/ConditionA/Rep1 \
  -o /vault/DatasetX/ConditionA/Rep1 \
  --canvas-name "DatasetX_ConditionA_Rep1" \
  --canvas-depth 3
```

> **Important:** Obsidian always opens the **vault root** as its working directory. The canvas file must therefore be somewhere inside the vault folder — `--canvas-depth` simply controls how many levels above `--output_dir` it ends up. Make sure `--output_dir` is nested deeply enough that going up `--canvas-depth` levels still lands inside the vault, not above it.

---

## Usage

### Basic Usage

```bash
python rel2obsi_beta.py -i /path/to/relion/project -o /path/to/output/notes
```

### Full Command-Line Reference

#### `rel2obsi_beta.py`

```
usage: rel2obsi_beta.py [-h] -i PROJECT_DIR -o OUTPUT_DIR [--force] [-v]
                        [--canvas-name CANVAS_NAME] [--canvas-depth CANVAS_DEPTH]

Generate Obsidian notes from RELION project jobs.

required arguments:
  -i, --project_dir     Path to the RELION project directory.
  -o, --output_dir      Path to the output directory for Obsidian notes.

optional arguments:
  -h, --help            Show this help message and exit.
  --force               Force regeneration of all existing notes.
  -v, --verbose         Enable verbose/debug logging.
  --canvas-name NAME    Name of the Canvas file (without .canvas extension).
                        Defaults to the RELION project directory name.
  --canvas-depth N      How many directory levels above --output_dir the Canvas
                        file is saved. Default: 2.
                        See "Vault Folder Structure" for examples.
```

#### `canvas_generator.py` (standalone)

The canvas generator can also be run independently if you only want to regenerate the job graph without recreating all notes:

```
usage: canvas_generator.py [-h] -i PROJECT_DIR -o OUTPUT_DIR [-v]
                            [--canvas-name CANVAS_NAME] [--canvas-depth CANVAS_DEPTH]

Creates an Obsidian Canvas for a RELION project.

required arguments:
  -i, --project_dir     Path to the RELION project directory.
  -o, --output_dir      Path to the Obsidian notes directory (where the .md files live).

optional arguments:
  -h, --help            Show this help message and exit.
  -v, --verbose         Enable verbose/debug logging.
  --canvas-name NAME    Name of the Canvas file (without .canvas extension).
                        Defaults to the RELION project directory name.
  --canvas-depth N      How many directory levels above --output_dir the Canvas
                        file is saved. Default: 2.
```

### Examples

Generate notes and canvas for a single project:
```bash
python rel2obsi_beta.py \
  -i ~/cryo_em/project \
  -o ~/obsidian_vault/project \
  --canvas-name "Project" \
  --canvas-depth 1 \
  --verbose
```

Regenerate only the canvas after layout changes (should not be necessary):
```bash
python canvas_generator.py \
  -i ~/cryo_em/project \
  -o ~/obsidian_vault/project \
  --canvas-name "Project" \
  --canvas-depth 1
```

Force-refresh all notes after a code update:
```bash
python rel2obsi_beta.py \
  -i ~/cryo_em/project \
  -o ~/obsidian_vault/project \
  --force
```

---

## Viewing the Results

1. Open Obsidian
2. Select **"Open folder as vault"**
3. Navigate to your **vault root** (the folder that contains the `.canvas` files and the project subfolders)
4. Start exploring from the `00_RELION_Project_Index.md` file, or open one of the `.canvas` files for a visual overview

---

## Understanding the Generated Output

### Note Structure

Each RELION job gets its own Markdown note with the following sections:

- **YAML frontmatter**: Job type, creation date, tags, and job-specific properties (resolution, particle counts, etc.)
- **Job Information**: Basic metadata
- **Input Jobs**: Wiki-links to upstream processing steps
- **Jobs that use this**: Wiki-links to downstream processing steps
- **Highlighted Settings**: Key parameters at a glance
- **All Settings**: Complete command-line parameters as a table
- **Job-specific sections**: Visual content depending on job type (see below)
- **RELION Command**: The full command used to run the job

### Job Type-Specific Content

| Job Type | Extra Content |
|----------|---------------|
| Class2D | Montage of 2D class averages with particle counts and estimated resolution per class |
| Class3D | Map slices (X/Y/Z) for the final iteration + orientation distribution histogram |
| Refine3D | Map slices + orientation histogram + resolution parsed from `run.out` |
| CtfFind | Linked `logfile.pdf` with CTF estimation plots |
| MotionCorr | Linked `logfile.pdf` with motion correction plots |
| Select | Particle or micrograph count parsed from `run.out` |
| Extract | Particle count parsed from `run.out` |
| Import | Item count parsed from `run.out` |

### Canvas Job Graph

The Canvas (`*.canvas`) file provides a visual overview of the entire processing pipeline:

- **Layout**: Tree structure growing top-to-bottom; each parent node is horizontally centered over its children
- **Columns (depth)**: Each row corresponds to one step in the processing chain
- **Colors**: Nodes are color-coded by job type (see table below); grey nodes indicate jobs that have not yet completed successfully
- **Links**: Clicking a node in Obsidian opens the corresponding Markdown note
- **Edges**: Arrows follow the data flow as recorded in `default_pipeline.star`

#### Node Colors

| Color | Job Types |
|-------|-----------|
| Green | Import |
| Yellow | MotionCorr, CtfFind, MaskCreate, LocalRes |
| Cyan | ManualPick, AutoPick |
| Orange | Extract, Select, Subset |
| Purple | Class2D, Class3D, InitialModel, MultiBody |
| Red | Refine3D, PostProcess, Polish, CtfRefine, BayesianPolishing |
| Grey | Any incomplete job |

---

## Troubleshooting

- **Missing dependencies**: Ensure all required Python packages are installed in the active environment
- **Canvas nodes not linked**: Verify that `default_pipeline.star` or `pipeline.star` exists in the RELION project root — this file is the sole source for job relationships in the canvas
- **Canvas saved in wrong location**: Check that `--canvas-depth` matches your actual folder nesting. Going up more levels than the vault depth will place the canvas outside the vault and Obsidian will not find it
- **File permissions**: Ensure write access to both `--output_dir` and the canvas target directory
- **Memory issues**: For very large projects, the 3D visualization steps are the most memory-intensive; try running with `--force` on a machine with more RAM or reduce the number of parallel workers in the source

For detailed error information, check `relion_to_obsidian.log` in the directory where you ran the script.

---

## Development Status

This tool is currently in active development. The beta version has limited testing — user feedback and bug reports are welcome.
