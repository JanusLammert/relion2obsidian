import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import mrcfile
import starfile
from skimage.util import montage, img_as_ubyte
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import re
import datetime
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("relion_to_obsidian.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rel2obsi")

def parse_relion_jobs(project_dir):
    """
    Parses the RELION project directory to extract job information.
    Optimized to reduce RAM usage by streaming file processing.

    :param project_dir: Path to the RELION project directory.
    :return: Generator yielding job dictionaries.
    """
    # Validate project directory exists
    if not os.path.isdir(project_dir):
        logger.error(f"Project directory does not exist: {project_dir}")
        raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
    
    job_count = 0
    error_count = 0
    
    # Walk through the directory structure
    for root, dirs, files in tqdm(os.walk(project_dir), desc="Scanning project directory"):
        for file in files:
            # Process only relevant files
            if file.endswith("job.star") or file.endswith("job.json"):
                job_path = os.path.join(root, file)
                job_name = os.path.basename(root)
                
                try:
                    # Determine job type based on folder name
                    job_type = next((t for t in [
                        "Class2D", "Class3D", "Refine3D", "Extract", "CtfFind",
                        "ManualPick", "AutoPick", "MaskCreate", "Micrographs", "MotionCorr",
                        "Select", "Import", "PostProcess"
                    ] if t in root), "Unknown")

                    # Get creation time for sorting/timeline
                    creation_time = os.path.getctime(job_path)
                    creation_date = datetime.datetime.fromtimestamp(creation_time)

                    # Parse job information (JSON or STAR)
                    job_details = {
                        "job_name": job_name, 
                        "job_path": os.path.abspath(job_path),
                        "creation_date": creation_date.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if file.endswith(".json"):
                        try:
                            with open(job_path, "r") as f:
                                job_json = json.load(f)
                                job_details.update(job_json)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON file {job_path}")
                        except Exception as e:
                            logger.warning(f"Error reading JSON file {job_path}: {str(e)}")

                    # Find input job relationships
                    input_jobs = []
                    pipeline_path = os.path.join(os.path.dirname(root), "pipeline.star")
                    if os.path.exists(pipeline_path):
                        try:
                            pipeline_data = starfile.read(pipeline_path)
                            if "pipeline_processes" in pipeline_data:
                                processes = pipeline_data["pipeline_processes"]
                                current_job_id = job_name.split("/")[-1] if "/" in job_name else job_name
                                
                                # Find the process for this job
                                job_row = processes[processes["rlnPipeLineProcessName"] == current_job_id]
                                if not job_row.empty:
                                    # Get input edges for this job from the process_edges table if available
                                    if "pipeline_input_edges" in pipeline_data:
                                        edges = pipeline_data["pipeline_input_edges"]
                                        # Get process ID for current job
                                        process_id = job_row["rlnPipeLineProcessID"].iloc[0]
                                        # Find input edges that connect to this process
                                        input_edges = edges[edges["rlnPipeLineProcessToID"] == process_id]
                                        
                                        # Get the corresponding input job names
                                        for _, edge in input_edges.iterrows():
                                            from_id = edge["rlnPipeLineProcessFromID"]
                                            from_job = processes[processes["rlnPipeLineProcessID"] == from_id]
                                            if not from_job.empty:
                                                input_job_name = from_job["rlnPipeLineProcessName"].iloc[0]
                                                input_jobs.append(input_job_name)
                        except Exception as e:
                            logger.warning(f"Error parsing pipeline.star for {job_name}: {str(e)}")

                    job_details["input_jobs"] = input_jobs

                    # Parse note.txt for command line parameters
                    note_path = os.path.join(root, "note.txt")
                    if os.path.exists(note_path):
                        try:
                            settings = {}
                            with open(note_path, "r") as note_file:
                                command_section = False
                                note_content = ""
                                for line in note_file:
                                    note_content += line
                                    line = line.strip()
                                    if "++++ with the following command(s):" in line:
                                        command_section = True
                                    elif command_section and "--" in line:
                                        parts = line.split()
                                        for i, part in enumerate(parts):
                                            if part.startswith("--"):
                                                key = part.lstrip("--")
                                                value = parts[i + 1] if (i + 1 < len(parts) and not parts[i + 1].startswith("--")) else "True"
                                                settings[key] = value
                            
                            job_details["settings"] = settings
                            
                            # Extract any user notes from the note.txt file
                            job_details["user_notes"] = note_content.strip()
                        except Exception as e:
                            logger.warning(f"Error parsing note.txt for {job_name}: {str(e)}")

                    # Add job-type specific tags
                    tags = ["relion", job_type.lower(), "cryo-em"]
                    
                    # Add additional tags based on job characteristics
                    if "Micrograph" in job_type or job_type == "Motion_Corr":
                        tags.append("micrograph-processing")
                    elif job_type in ["Class2D", "Class3D"]:
                        tags.append("classification")
                    elif job_type == "Refine3D":
                        tags.append("refinement")
                    elif job_type == "Extract":
                        tags.append("particle-extraction")
                    elif job_type in ["ManualPick", "AutoPick"]:
                        tags.append("picking")
                    
                    # Add resolution tag for specific job types
                    if job_type in ["Refine3D", "PostProcess"]:
                        if "settings" in job_details and "angpix" in job_details["settings"]:
                            tags.append("resolution")
                    
                    job_details["tags"] = tags
                    
                    job_count += 1
                    yield {"name": job_name, "details": job_details, "type": job_type}
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing job {job_name}: {str(e)}")
                    logger.debug(traceback.format_exc())
    
    logger.info(f"Processed {job_count} jobs with {error_count} errors")

def normalize(img):
    """Normalize image values to [0,1] range"""
    try:
        min_val, max_val = np.min(img), np.max(img)
        return img if min_val == max_val else (img - min_val) / (max_val - min_val)
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        return img  # Return original image if normalization fails

def add_text(gray_image_array, top_text="Top Text", bottom_text="Bottom Text"):
    """Add text to an image"""
    try:
        gray_image_array = normalize(gray_image_array)
        gray_image_array = img_as_ubyte(gray_image_array)

        image = Image.fromarray(gray_image_array).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        text_color = (0, 255, 0)

        image_width, image_height = image.size

        # Calculate the bounding box for the top text
        top_text_bbox = draw.textbbox((0, 0), top_text, font=font)
        text_width = top_text_bbox[2] - top_text_bbox[0]
        text_height = top_text_bbox[3] - top_text_bbox[1]

        top_text_position = ((image_width - text_width) // 2, 10)
        draw.text(top_text_position, top_text, fill=text_color, font=font)

        # Calculate the bounding box for the bottom text
        bottom_text_bbox = draw.textbbox((0, 0), bottom_text, font=font)
        text_width = bottom_text_bbox[2] - bottom_text_bbox[0]
        text_height = bottom_text_bbox[3] - bottom_text_bbox[1]

        bottom_text_position = ((image_width - text_width) // 2, image_height - text_height - 10)
        draw.text(bottom_text_position, bottom_text, fill=text_color, font=font)

        return np.array(image)
        
    except Exception as e:
        logger.error(f"Error adding text to image: {str(e)}")
        return gray_image_array  # Return original image if text addition fails

def make_montage(stk, ncols=10):
    """Create a montage of images"""
    try:
        nrows = int(np.ceil(stk.shape[0] / ncols))
        return montage(stk, grid_shape=(nrows, ncols), channel_axis=-1)
    except Exception as e:
        logger.error(f"Error creating montage: {str(e)}")
        return None

def generate_class2d_image(job_dir, out_dir, iteration=-1):
    """Generate montage image from Class2D job results"""
    try:
        job_dir = Path(job_dir)
        # If job_dir is a file path, get the parent directory
        if job_dir.is_file():
            job_dir = job_dir.parent
            
        class2d_dir = job_dir
        job_nr = job_dir.name

        # Find necessary files
        model_star_files = sorted(class2d_dir.glob("*model.star"))
        particle_star_files = sorted(class2d_dir.glob("*data.star"))
        mrc_stack_files = sorted(class2d_dir.glob("*classes.mrcs"))

        if not model_star_files or not particle_star_files or not mrc_stack_files:
            logger.warning(f"Missing required files for Class2D visualization in {job_dir}")
            return None

        iteration = iteration if iteration >= 0 else min(
            len(model_star_files), len(particle_star_files), len(mrc_stack_files)
        ) - 1

        # Read only required columns to save memory
        try:
            model_data = starfile.read(model_star_files[iteration])
            particles_data = starfile.read(particle_star_files[iteration])
            
            with mrcfile.open(mrc_stack_files[iteration], permissive=True) as f:
                mrc_stk = f.data
        except Exception as e:
            logger.error(f"Error reading Class2D data files: {str(e)}")
            return None

        # Reduce dataframe and convert to numpy to save memory
        classes_df = model_data["model_classes"]
        total_particles = particles_data["particles"].shape[0]

        # Select required columns and convert to numpy arrays
        class_distribution = classes_df["rlnClassDistribution"].to_numpy()
        estimated_resolution = classes_df["rlnEstimatedResolution"].to_numpy()
        number_of_particles = (class_distribution * total_particles).astype(int)

        # Sort indices by descending class distribution
        sorted_indices = np.argsort(-class_distribution)

        # Sort mrc stack incrementally to avoid creating a large sorted copy
        mrc_stk_sorted = np.empty_like(mrc_stk)
        for i, idx in enumerate(sorted_indices):
            mrc_stk_sorted[i] = mrc_stk[idx]

        # Generate labeled images incrementally
        labeled_images = []
        for img, num, res in zip(
            mrc_stk_sorted, number_of_particles[sorted_indices], estimated_resolution[sorted_indices]
        ):
            labeled_images.append(add_text(img, str(num), f"{res:.2f} A"))

        # Stack labeled images and create a montage
        montage_img = make_montage(np.stack(labeled_images))
        if montage_img is None:
            return None

        # Save the montage to a file
        rel_montage_path = f"assets/Class2D_{job_nr}_montage_It_{iteration}.png"
        output_path = Path(out_dir) / rel_montage_path
        
        # Create assets directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.imsave(output_path, montage_img, cmap="gray")
        logger.info(f"Created Class2D montage: {output_path}")

        return rel_montage_path

    except Exception as e:
        logger.error(f"Error generating montage for {job_dir}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def get_job_note_filename(job):
    """Generate consistent filename for a job"""
    # Sanitize filename to avoid issues in different file systems
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", job['name'])
    return f"{job['type']}_{safe_name}.md"

def update_note_with_backward_link(note_path, new_job_filename, new_job_name, job_type):
    """Update an existing note with a link to a job that uses it as input"""
    if not os.path.exists(note_path):
        logger.warning(f"Can't update non-existent note: {note_path}")
        return False
        
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the Jobs that use this section exists
        if "## Jobs that use this" not in content:
            # Add the section if it doesn't exist
            content += "\n\n## Jobs that use this\n"
        
        # Add the link if it doesn't already exist
        link_text = f"- [[{new_job_filename.replace('.md', '')}|{job_type}: {new_job_name}]]"
        if link_text not in content:
            # Find the section and append the link
            sections = re.split(r'(?=^## )', content, flags=re.MULTILINE)
            for i, section in enumerate(sections):
                if section.startswith("## Jobs that use this"):
                    if section.strip() == "## Jobs that use this":
                        # If the section is empty
                        sections[i] = f"## Jobs that use this\n{link_text}\n"
                    else:
                        # Append to existing content
                        sections[i] = f"{section.rstrip()}\n{link_text}\n"
                    break
            
            # If we didn't find the section, add it
            if not any(s.startswith("## Jobs that use this") for s in sections):
                sections.append(f"## Jobs that use this\n{link_text}\n")
            
            # Join the sections back together
            content = ''.join(sections)
            
            with open(note_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.debug(f"Updated {note_path} with backward link to {new_job_filename}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating note with backward link {note_path}: {str(e)}")
        return False

def create_obsidian_notes(jobs, output_dir, force=False):
    """Create or update Obsidian notes for all jobs"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create assets directory for images
        assets_dir = os.path.join(output_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Dictionary to store all jobs for quick lookups
        all_jobs = {}
        for job in jobs:
            all_jobs[job['name']] = job
        
        # First, create all notes to ensure they exist for cross-linking
        logger.info(f"Creating notes for {len(all_jobs)} jobs...")
        for job in tqdm(all_jobs.values(), desc="Creating notes"):
            note_filename = get_job_note_filename(job)
            note_path = os.path.join(output_dir, note_filename)
            
            # Check if the note already exists and we're not forcing recreation
            if os.path.exists(note_path) and not force:
                continue
                
            try:
                with open(note_path, "w", encoding='utf-8') as f:
                    # Add YAML frontmatter
                    f.write("---\n")
                    f.write(f"title: {job['type']} - {job['name']}\n")
                    f.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
                    
                    # Add tags from job details or defaults
                    tags = job['details'].get('tags', ["relion", job['type'].lower(), "cryo-em"])
                    # Convert tags list to string
                    tags_str = ', '.join([f'"{tag}"' for tag in tags])
                    f.write(f"tags: [{tags_str}]\n")
                    
                    # Add creation date from job details if available
                    creation_date = job['details'].get('creation_date', '')
                    if creation_date:
                        f.write(f"creation_date: {creation_date}\n")

                    f.write("---\n\n")
                    
                    # Job header
                    f.write(f"# {job['type']}: {job['name']}\n\n")
                    
                    # Basic info section
                    f.write("## Job Information\n\n")
                    f.write(f"- **Job Type**: {job['type']}\n")
                    f.write(f"- **Job Name**: {job['name']}\n")
                    f.write(f"- **Path**: `{job['details'].get('job_path', '')}`\n")
                    
                    if 'creation_date' in job['details']:
                        f.write(f"- **Created**: {job['details']['creation_date']}\n")
                    
                    f.write("\n")
                    
                    # Input jobs section if applicable
                    input_jobs = job['details'].get('input_jobs', [])
                    if input_jobs:
                        f.write("## Input Jobs\n\n")
                        for input_job_name in input_jobs:
                            # Create a link to the input job
                            if input_job_name in all_jobs:
                                input_job = all_jobs[input_job_name]
                                input_filename = get_job_note_filename(input_job)
                                link_text = f"{input_job['type']}: {input_job_name}"
                                f.write(f"- [[{input_filename.replace('.md', '')}|{link_text}]]\n")
                            else:
                                f.write(f"- {input_job_name} (not found)\n")
                        f.write("\n")
                    
                    # Placeholder for jobs that use this job
                    f.write("## Jobs that use this\n\n")
                    
                    # User notes section if available
                    if 'user_notes' in job['details'] and job['details']['user_notes'].strip():
                        f.write("## User Notes\n\n")
                        f.write("```\n")
                        f.write(job['details']['user_notes'])
                        f.write("\n```\n\n")
                    
                    # Details section
                    f.write("## Additional Details\n\n")
                    for key, value in job["details"].items():
                        if key not in ["settings", "input_jobs", "job_name", "job_path", "user_notes", "tags", "creation_date"]:
                            f.write(f"- **{key}**: {value}\n")
                    f.write("\n")

                    # Highlighted settings section
                    highlighted_settings = [
                        "o", "iter", "blush", "sym", "helical_twist_initial",
                        "helical_rise_initial", "helical_symmetry_search", "solvent_mask"
                    ]
                    
                    highlighted = {
                        k: v for k, v in job["details"].get("settings", {}).items() if k in highlighted_settings
                    }
                    if highlighted:
                        f.write("## Highlighted Settings\n\n")
                        f.write("| Setting | Value |\n")
                        f.write("|---------|-------|\n")
                        for key, value in highlighted.items():
                            f.write(f"| `{key}` | `{value}` |\n")
                        f.write("\n")

                    # All settings in a collapsible section
                    if "settings" in job["details"] and job["details"]["settings"]:
                        f.write("## All Settings\n\n")
                        f.write("<details>\n")
                        f.write("<summary>Click to expand all settings</summary>\n\n")
                        f.write("| Setting | Value |\n")
                        f.write("|---------|-------|\n")
                        for key, value in job["details"]["settings"].items():
                            # Format value to handle long values
                            value_str = str(value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                            f.write(f"| `{key}` | `{value_str}` |\n")
                        f.write("\n</details>\n\n")

                    # Special sections for specific job types
                    if job['type'] == "Class2D":
                        f.write("## 2D Class Averages\n\n")
                        montage_path = generate_class2d_image(job["details"].get("job_path", ""), output_dir)
                        if montage_path:
                            f.write(f"![Class2D Montage]({montage_path})\n\n")
                        else:
                            f.write("*Class averages could not be generated*\n\n")
                    
                    elif job['type'] == "Class3D":
                        f.write("## 3D Classification\n\n")
                        f.write("*Images of 3D classes would be shown here when available*\n\n")
                    
                    elif job['type'] == "Refine3D":
                        f.write("## 3D Refinement Results\n\n")
                        f.write("*Refinement statistics and FSC curves would be shown here when available*\n\n")
                        
                        # Find the PostProcess job that might be associated with this refinement
                        # This is based on naming pattern in RELION
                        possible_postproc = [
                            j for j in all_jobs.values() 
                            if j['type'] == 'PostProcess' and 
                            j['details'].get('input_jobs', []) and 
                            job['name'] in j['details']['input_jobs']
                        ]
                        
                        if possible_postproc:
                            f.write("## Associated PostProcess Jobs\n\n")
                            for pp_job in possible_postproc:
                                pp_filename = get_job_note_filename(pp_job)
                                f.write(f"- [[{pp_filename.replace('.md', '')}|PostProcess: {pp_job['name']}]]\n")
                            f.write("\n")
            
            except Exception as e:
                logger.error(f"Error creating note for {job['name']}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        # Now update notes with backward links
        logger.info("Creating backward links between jobs...")
        for job in tqdm(all_jobs.values(), desc="Creating backward links"):
            input_jobs = job['details'].get('input_jobs', [])
            note_filename = get_job_note_filename(job)
            
            for input_job_name in input_jobs:
                if input_job_name in all_jobs:
                    input_job = all_jobs[input_job_name]
                    input_note_path = os.path.join(output_dir, get_job_note_filename(input_job))
                    update_note_with_backward_link(input_note_path, note_filename, job['name'], job['type'])
        
        # Create an index file for easier navigation
        create_index_file(all_jobs, output_dir)
        
    except Exception as e:
        logger.error(f"Error creating Obsidian notes: {str(e)}")
        logger.debug(traceback.format_exc())

def create_index_file(all_jobs, output_dir):
    """Create an index file with links to all jobs, organized by job type"""
    try:
        index_path = os.path.join(output_dir, "00_RELION_Project_Index.md")
        
        with open(index_path, "w", encoding='utf-8') as f:
            f.write("---\n")
            f.write("title: RELION Project Index\n")
            f.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("tags: [relion, index, cryo-em]\n")
            f.write("---\n\n")
            
            f.write("# RELION Project Index\n\n")
            f.write("This is an automatically generated index of all RELION jobs in this project.\n\n")
            
            # Group jobs by type
            job_types = {}
            for job in all_jobs.values():
                job_type = job['type']
                if job_type not in job_types:
                    job_types[job_type] = []
                job_types[job_type].append(job)
            
            # Add table of contents
            f.write("## Job Types\n\n")
            for job_type in sorted(job_types.keys()):
                f.write(f"- [{job_type}](#{job_type.lower()})\n")
            f.write("\n")
            
            # Add sections for each job type
            for job_type in sorted(job_types.keys()):
                f.write(f"## {job_type}\n\n")
                
                # Sort jobs by name
                jobs = sorted(job_types[job_type], key=lambda j: j['name'])
                
                f.write("| Job Name | Created | Input Jobs |\n")
                f.write("|----------|---------|------------|\n")
                
                for job in jobs:
                    note_filename = get_job_note_filename(job)
                    creation_date = job['details'].get('creation_date', '')
                    
                    # Get input job count
                    input_jobs = job['details'].get('input_jobs', [])
                    input_job_count = len(input_jobs)
                    
                    f.write(f"| [[{note_filename.replace('.md', '')}|{job['name']}]] | {creation_date} | {input_job_count} |\n")
                
                f.write("\n")
            
            # Add tag section for filtering
            f.write("## Tags\n\n")
            
            # Collect all unique tags
            all_tags = set()
            for job in all_jobs.values():
                all_tags.update(job['details'].get('tags', []))
            
            # Write tag list
            for tag in sorted(all_tags):
                f.write(f"- #{tag}\n")
            
        logger.info(f"Created index file: {index_path}")
    except Exception as e:
        logger.error(f"Error creating index file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate Obsidian notes from RELION project jobs.")
    parser.add_argument("-i", "--project_dir", required=True, help="Path to the RELION project directory.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory for Obsidian notes.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of existing notes.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Show script information
        logger.info(f"Relion to Obsidian Converter")
        logger.info(f"Project directory: {args.project_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Read all job data first to build the complete relationship graph
        logger.info("Parsing RELION project directory...")
        jobs = list(parse_relion_jobs(args.project_dir))
        logger.info(f"Found {len(jobs)} jobs.")
        
        # Create or update notes with links between jobs
        create_obsidian_notes(jobs, args.output_dir, args.force)
        logger.info(f"Obsidian notes created in {args.output_dir}")
        logger.info("Open this directory as an Obsidian vault to view the job structure.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
