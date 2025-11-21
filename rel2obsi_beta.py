import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from natsort import natsorted
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import matplotlib
matplotlib.use('Agg')

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

def find_job_files(project_dir, max_depth=2, current_depth=0):
    """Fast scan for job files using os.scandir, limited to max_depth.
    At the top level, only recurse into known job-type directories.
    Always skip Micrographs folders.
    """
    job_files = []
    job_dir_patterns = [
        'Import', 'MotionCorr', 'CtfFind', 'ManualPick', 'AutoPick', 'Extract', 'Select', 'Subset',
        'Class2D', 'Class3D', 'Refine3D', 'InitialModel', 'MultiBody', 'Reconstruct',
        'Polish', 'CtfRefine', 'BayesianPolishing', 'PostProcess', 'LocalRes', 'MaskCreate',
        'External', 'Subtract', 'JoinStar', 'Split', 'MovieRefine', 'TiltSeries',
        'TomogramReconstruct', 'TomogramCtfRefine', 'TomogramClassify3D', 'TomogramRefine3D',
        'ResMap', 'MultiBodyRefine', 'HelicalRefine3D', 'HelicalInitialModel',
        'Export', 'ImportMovies', 'MotionCorrMulti', 'Recenter', 'RelionIt',
        'CoordinateExport', 'CtfPlot', 'ParticleSubtract', 'AutoRefine'
    ]

    try:
        for entry in os.scandir(project_dir):
            if entry.is_dir(follow_symlinks=False):
                # Micrographs oder micrographs komplett überspringen
                if entry.name.lower() == "micrographs":
                    continue

                # Nur im ersten Level (current_depth == 0) filtern nach bekannten Jobtypen
                if current_depth == 0:
                    if entry.name not in job_dir_patterns:
                        continue  # Nicht in Liste → überspringen

                # In erlaubte Ordner weitergehen
                if current_depth < max_depth:
                    job_files.extend(find_job_files(entry.path, max_depth, current_depth + 1))

            elif entry.is_file() and (entry.name.endswith("job.star") or entry.name.endswith("job.json")):
                job_files.append(entry.path)

    except PermissionError:
        logger.warning(f"Permission denied: {project_dir}")
    return job_files


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
    
# Cache pipeline data to avoid re-reading for each job
    pipeline_cache = {}
    
    # Fast scan for all job files first
    logger.info("Scanning for job files...")
    job_files = find_job_files(project_dir)
    logger.info(f"Found {len(job_files)} job files to process")
    
    # Process job files
    for job_path in tqdm(job_files, desc="Processing jobs"):
        job_name = os.path.basename(os.path.dirname(job_path))
        root = os.path.dirname(job_path)
        file = os.path.basename(job_path)
        
        try:
            # Determine job type based on folder name
            job_dir_patterns = [
                'Import', 'MotionCorr', 'CtfFind', 'ManualPick', 'AutoPick', 'Extract', 'Select', 'Subset',
                'Class2D', 'Class3D', 'Refine3D', 'InitialModel', 'MultiBody', 'Reconstruct',
                'Polish', 'CtfRefine', 'BayesianPolishing', 'PostProcess', 'LocalRes', 'MaskCreate',
                'External', 'Subtract', 'JoinStar', 'Split', 'MovieRefine', 'TiltSeries',
                'TomogramReconstruct', 'TomogramCtfRefine', 'TomogramClassify3D', 'TomogramRefine3D',
                'ResMap', 'MultiBodyRefine', 'HelicalRefine3D', 'HelicalInitialModel',
                'Export', 'ImportMovies', 'MotionCorrMulti', 'Recenter', 'RelionIt',
                'CoordinateExport', 'CtfPlot', 'ParticleSubtract', 'AutoRefine'
            ]

            job_type = next((t for t in job_dir_patterns if t in root), "Unknown")


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

            # --- Find input/output job relationships ---
            input_jobs = []
            output_jobs = []

            # --- Locate and load default_pipeline.star or pipeline.star ---
            possible_pipelines = [
                os.path.join(project_dir, "default_pipeline.star"),
                os.path.join(project_dir, "pipeline.star"),
            ]
            pipeline_path = next((p for p in possible_pipelines if os.path.exists(p)), None)

            if pipeline_path not in pipeline_cache:
                if pipeline_path:
                    try:
                        pipeline_cache[pipeline_path] = starfile.read(pipeline_path)
                        logger.debug(f"Cached pipeline file: {pipeline_path}")
                    except Exception as e:
                        logger.warning(f"Error reading pipeline file {pipeline_path}: {str(e)}")
                        pipeline_cache[pipeline_path] = None
                else:
                    pipeline_cache[project_dir] = None

            pipeline_data = pipeline_cache.get(pipeline_path)

            # --- Parse edges directly from your default_pipeline.star structure ---
            if isinstance(pipeline_data, dict):
                processes = (
                    pipeline_data.get("data_pipeline_processes")
                    or pipeline_data.get("pipeline_processes")
                )
                edges = (
                    pipeline_data.get("data_pipeline_input_edges")
                    or pipeline_data.get("pipeline_input_edges")
                )

                if processes is not None and edges is not None:
                    try:
                        # current job name, e.g. 'job003'
                        current_job_id = job_name.split("/")[-1]
                        # full process name, e.g. 'CtfFind/job003/' (with trailing slash)
                        current_proc_full = processes[
                            processes["rlnPipeLineProcessName"].str.endswith(current_job_id + "/")
                        ]
                        if not current_proc_full.empty:
                            process_name = current_proc_full["rlnPipeLineProcessName"].iloc[0]

                            # 1️⃣ find all edges that *point to* this job (inputs)
                            for _, edge in edges.iterrows():
                                to_proc = edge["rlnPipeLineEdgeProcess"].strip()
                                if to_proc.rstrip("/") == process_name.rstrip("/"):
                                    from_node = edge["rlnPipeLineEdgeFromNode"].strip()
                                    # Extract upstream job like 'MotionCorr/job002'
                                    parts = from_node.split("/")
                                    if len(parts) >= 2:
                                        from_job = f"{parts[0]}/{parts[1]}"
                                        if from_job not in input_jobs:
                                            input_jobs.append(from_job)

                            # 2️⃣ find all edges that *originate* from this job (outputs)
                            for _, edge in edges.iterrows():
                                from_node = edge["rlnPipeLineEdgeFromNode"].strip()
                                if from_node.startswith(process_name.rstrip("/")):
                                    to_proc = edge["rlnPipeLineEdgeProcess"].strip()
                                    parts = to_proc.split("/")
                                    if len(parts) >= 2:
                                        to_job = f"{parts[0]}/{parts[1]}"
                                        if to_job not in output_jobs:
                                            output_jobs.append(to_job)

                        logger.debug(
                            f"Job {job_name}: inputs={input_jobs}, outputs={output_jobs}"
                        )

                    except Exception as e:
                        logger.warning(f"Error parsing default_pipeline.star for {job_name}: {str(e)}")

            job_details["input_jobs"] = input_jobs
            job_details["output_jobs"] = output_jobs


            
            # Use cached pipeline data if available
            if pipeline_path:
                if pipeline_path not in pipeline_cache:
                    try:
                        pipeline_cache[pipeline_path] = starfile.read(pipeline_path)
                        logger.debug(f"Cached pipeline file: {pipeline_path}")
                    except Exception as e:
                        logger.warning(f"Error reading pipeline file {pipeline_path}: {str(e)}")
                        pipeline_cache[pipeline_path] = None
            else:
                logger.warning("No pipeline.star or default_pipeline.star found in project root")
                pipeline_cache[project_dir] = None
            
            # pipeline_data = pipeline_cache.get(pipeline_path)
            # if pipeline_data and "pipeline_processes" in pipeline_data:
            #     try:
            #         processes = pipeline_data["pipeline_processes"]
            #         current_job_id = job_name.split("/")[-1]
            #         job_row = processes[processes["rlnPipeLineProcessName"].str.endswith(current_job_id)]
            #         if not job_row.empty:
            #             # Get input edges for this job from the process_edges table if available
            #             if "pipeline_input_edges" in pipeline_data:
            #                 edges = pipeline_data["pipeline_input_edges"]
            #                 # Get process ID for current job
            #                 process_id = job_row["rlnPipeLineProcessID"].iloc[0]
            #                 # Find input edges that connect to this process
            #                 input_edges = edges[edges["rlnPipeLineProcessToID"] == process_id]
                            
            #                 # Get the corresponding input job names
            #                 for _, edge in input_edges.iterrows():
            #                     from_id = edge["rlnPipeLineProcessFromID"]
            #                     from_job = processes[processes["rlnPipeLineProcessID"] == from_id]
            #                     if not from_job.empty:
            #                         input_job_name = from_job["rlnPipeLineProcessName"].iloc[0]
            #                         input_jobs.append(input_job_name)
            #                 logger.debug(f"Job {job_name} - found {len(input_jobs)} input jobs: {input_jobs}")

            #     except Exception as e:
            #         logger.warning(f"Error parsing pipeline.star for {job_name}: {str(e)}")

            # job_details["input_jobs"] = input_jobs

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


        #iteration = iteration if iteration >= 0 else min(
        #    len(model_star_files), len(particle_star_files), len(mrc_stack_files)
        #) - 1


        # Check if job is still running by looking for a RELION_JOB_EXIT_SUCCESS file
        success_file = class2d_dir / "RELION_JOB_EXIT_SUCCESS"
        job_is_incomplete = not success_file.exists()
        
        if job_is_incomplete:
            logger.warning(f"Class2D job {job_nr} appears to be still running or incomplete - will create visualization anyway")
        
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
    
def generate_refine3d_visualization(job_dir, out_dir, job_name):
    """Generate visualization for Refine3D job (map slices + orientation histogram)"""
    try:
        job_dir = Path(job_dir)
        if job_dir.is_file():
            job_dir = job_dir.parent
        
        # Check if job is complete
        success_file = job_dir / "RELION_JOB_EXIT_SUCCESS"
        if not success_file.exists():
            logger.debug(f"Refine3D job {job_name} not yet complete - skipping visualization")
            return None
        
        job_nr = job_dir.name
        
        # Find required files
        filtered_maps = list(job_dir.glob("run_class00?.mrc"))
        particles_star_fpath = job_dir / "run_data.star"
        model_star_fpath = job_dir / "run_model.star"
        
        if not filtered_maps or not particles_star_fpath.exists() or not model_star_fpath.exists():
            logger.warning(f"Missing required files for Refine3D visualization in {job_dir}")
            return None
        
        NUM_CLASSES = len(filtered_maps)
        
        # Read map dimensions
        with mrcfile.open(filtered_maps[0]) as mrc:
            MAP_SHAPE = mrc.data.shape
        
        # Load particle data
        particle_df = starfile.read(particles_star_fpath)["particles"]
        
        # Load model data for resolution
        model_dict = starfile.read(model_star_fpath)
        fsc_resolution = model_dict["model_general"]["rlnCurrentResolution"]
        
        # Create combined figure
        total_rows = NUM_CLASSES + 1
        fig = plt.figure(figsize=(15, 5 * total_rows))
        
        # Plot map slices
        for i, map_file in enumerate(filtered_maps):
            with mrcfile.open(map_file) as mrc:
                map_arr = mrc.data
                map_arr = normalize(map_arr)
            
            ax1 = plt.subplot(total_rows, 3, i*3 + 1)
            ax2 = plt.subplot(total_rows, 3, i*3 + 2) 
            ax3 = plt.subplot(total_rows, 3, i*3 + 3)
            
            mid_z = map_arr.shape[0] // 2
            mid_y = map_arr.shape[1] // 2
            mid_x = map_arr.shape[2] // 2

            ax1.imshow(map_arr[mid_z, :, :], cmap='gray')
            ax2.imshow(map_arr[:, mid_y, :], cmap='gray')
            ax3.imshow(map_arr[:, :, mid_x], cmap='gray')

            ax1.set_title(f"Class {i+1} - Z slice")
            ax2.set_title(f"Class {i+1} - Y slice")
            ax3.set_title(f"Class {i+1} - X slice")

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
        
        # Orientation histogram
        ax_orient = plt.subplot(total_rows, 1, total_rows)
        
        azimuthal_angles = np.array(particle_df["rlnAngleRot"])
        polar_angles = np.array(particle_df["rlnAngleTilt"])
        
        # Auto bin calculation using Sturges
        nbins_x = int(np.floor(np.log2(len(azimuthal_angles)) + 1))
        nbins_y = int(np.floor(np.log2(len(polar_angles)) + 1))
        bins = (nbins_x, nbins_y)
        
        h = ax_orient.hist2d(azimuthal_angles, polar_angles, bins=bins, cmap='viridis')
        
        cbar = plt.colorbar(h[3], ax=ax_orient)
        cbar.set_label('Count')
        
        ax_orient.set_xlabel('Azimuthal Angle ("Rot") (φ)°')
        ax_orient.set_ylabel('Polar Angle ("Tilt") (θ)°')
        ax_orient.set_title(f'Orientation Distribution')
        ax_orient.grid(alpha=0.3)
        
        fig.suptitle(f"Refine3D Analysis: {job_name} ({fsc_resolution:.2f} Å)", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        rel_output_path = f"assets/Refine3D_{job_nr}_analysis.png"
        output_path = Path(out_dir) / rel_output_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created Refine3D visualization: {output_path}")
        return rel_output_path
        
    except Exception as e:
        logger.error(f"Error generating Refine3D visualization for {job_dir}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None
    
def extract_iteration_number(filename):
    """Extract the iteration number from a filename like 'run_it050_class001.mrc'"""
    base_filename = os.path.basename(filename)
    pattern = r'it(\d+)'
    match = re.search(pattern, base_filename)
    
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No iteration number found in filename: {filename}")

def extract_iteration_number_as_string(filename):
    """Extract the iteration number from a filename, preserving leading zeros"""
    base_filename = os.path.basename(filename)
    pattern = r'it(\d+)'
    match = re.search(pattern, base_filename)
    
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No iteration number found in filename: {filename}")

def generate_class3d_visualization(job_dir, out_dir, job_name):
    """Generate visualization for Class3D job (map slices + orientation histogram)"""
    try:
        job_dir = Path(job_dir)
        if job_dir.is_file():
            job_dir = job_dir.parent
        
        # Check if job is complete
        success_file = job_dir / "RELION_JOB_EXIT_SUCCESS"
        if not success_file.exists():
            logger.debug(f"Class3D job {job_name} not yet complete - skipping visualization")
            return None
        
        job_nr = job_dir.name
        
        # Find filtered maps with iteration pattern
        map_files = list(job_dir.glob("run_it???_class00?.mrc"))
        if not map_files:
            logger.warning(f"No class maps found for Class3D job {job_name}")
            return None
        
        map_files = natsorted(map_files)
        
        # Get iteration from last map (most recent iteration)
        iter_num = extract_iteration_number(map_files[-1].name)
        iter_num_string = extract_iteration_number_as_string(map_files[-1].name)
        
        # Get all maps from this iteration
        map_files_one_it = list(job_dir.glob(f"*{iter_num_string}*.mrc"))
        
        if not map_files_one_it:
            logger.warning(f"No maps found for iteration {iter_num_string} in {job_dir}")
            return None
        
        NUM_CLASSES = len(map_files_one_it)
        
        # Read map dimensions
        with mrcfile.open(map_files_one_it[0]) as mrc:
            MAP_SHAPE = mrc.data.shape
        
        # Load particle data for this iteration
        particles_star_fpath = job_dir / f"run_it{iter_num_string}_data.star"
        model_star_fpath = job_dir / f"run_it{iter_num_string}_model.star"
        
        if not particles_star_fpath.exists() or not model_star_fpath.exists():
            logger.warning(f"Missing required files for Class3D iteration {iter_num_string} in {job_dir}")
            return None
        
        particle_df = starfile.read(particles_star_fpath)["particles"]
        model_dict = starfile.read(model_star_fpath)
        est_resolution = model_dict["model_general"]["rlnCurrentResolution"]
        
        # Create combined figure
        total_rows = NUM_CLASSES + 1
        fig = plt.figure(figsize=(15, 5 * total_rows))
        
        # Plot map slices
        for i, map_file in enumerate(map_files_one_it):
            with mrcfile.open(map_file) as mrc:
                map_arr = mrc.data
                map_arr = normalize(map_arr)
            
            ax1 = plt.subplot(total_rows, 3, i*3 + 1)
            ax2 = plt.subplot(total_rows, 3, i*3 + 2) 
            ax3 = plt.subplot(total_rows, 3, i*3 + 3)
            
            mid_z = map_arr.shape[0] // 2
            mid_y = map_arr.shape[1] // 2
            mid_x = map_arr.shape[2] // 2

            ax1.imshow(map_arr[mid_z, :, :], cmap='gray')
            ax2.imshow(map_arr[:, mid_y, :], cmap='gray')
            ax3.imshow(map_arr[:, :, mid_x], cmap='gray')

            ax1.set_title(f"Class {i+1} - Z slice")
            ax2.set_title(f"Class {i+1} - Y slice")
            ax3.set_title(f"Class {i+1} - X slice")

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
        
        # Orientation histogram
        ax_orient = plt.subplot(total_rows, 1, total_rows)
        
        azimuthal_angles = np.array(particle_df["rlnAngleRot"])
        polar_angles = np.array(particle_df["rlnAngleTilt"])
        
        # Auto bin calculation using Sturges
        nbins_x = int(np.floor(np.log2(len(azimuthal_angles)) + 1))
        nbins_y = int(np.floor(np.log2(len(polar_angles)) + 1))
        bins = (nbins_x, nbins_y)
        
        h = ax_orient.hist2d(azimuthal_angles, polar_angles, bins=bins, cmap='viridis')
        
        cbar = plt.colorbar(h[3], ax=ax_orient)
        cbar.set_label('Count')
        
        ax_orient.set_xlabel('Azimuthal Angle ("Rot") (φ)°')
        ax_orient.set_ylabel('Polar Angle ("Tilt") (θ)°')
        ax_orient.set_title(f'Orientation Distribution - {particles_star_fpath.name}')
        ax_orient.grid(alpha=0.3)
        
        fig.suptitle(f"Class3D Analysis: {job_name} It. {iter_num} ({est_resolution:.2f} Å)", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        rel_output_path = f"assets/Class3D_{job_nr}_it{iter_num_string}_analysis.png"
        output_path = Path(out_dir) / rel_output_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created Class3D visualization: {output_path}")
        return rel_output_path
        
    except Exception as e:
        logger.error(f"Error generating Class3D visualization for {job_dir}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None
    
def batch_generate_refine3d_visualizations(refine3d_jobs, output_dir, force=False, max_workers=2):
    """Generate Refine3D visualizations in parallel"""
    refine3d_visualizations = {}
    
    def process_single_refine3d(job):
        """Process a single Refine3D job and return its visualization path"""
        try:
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            job_nr = job_dir.name
            
            expected_vis_path = f"assets/Refine3D_{job_nr}_analysis.png"
            full_vis_path = Path(output_dir) / expected_vis_path
            
            if full_vis_path.exists() and not force:
                logger.debug(f"Refine3D visualization already exists: {expected_vis_path}")
                return job['name'], expected_vis_path
            
            # Generate new visualization
            vis_path = generate_refine3d_visualization(
                job["details"].get("job_path", ""), 
                output_dir, 
                job['name']
            )
            return job['name'], vis_path
            
        except Exception as e:
            logger.error(f"Error in parallel processing of Refine3D {job['name']}: {str(e)}")
            return job['name'], None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_refine3d, job): job for job in refine3d_jobs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Refine3D visualizations"):
            try:
                job_name, vis_path = future.result()
                if vis_path:
                    refine3d_visualizations[job_name] = vis_path
            except Exception as e:
                job = futures[future]
                logger.error(f"Error processing Refine3D job {job['name']}: {str(e)}")
    
    return refine3d_visualizations

def batch_generate_class3d_visualizations(class3d_jobs, output_dir, force=False, max_workers=2):
    """Generate Class3D visualizations in parallel"""
    class3d_visualizations = {}
    
    def process_single_class3d(job):
        """Process a single Class3D job and return its visualization path"""
        try:
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            job_nr = job_dir.name
            
            # Find the most recent iteration to determine expected output filename
            map_files = list(job_dir.glob("run_it???_class00?.mrc"))
            if not map_files:
                return job['name'], None
            
            map_files = natsorted(map_files)
            try:
                iter_num_string = extract_iteration_number_as_string(map_files[-1].name)
            except ValueError:
                return job['name'], None
            
            expected_vis_path = f"assets/Class3D_{job_nr}_it{iter_num_string}_analysis.png"
            full_vis_path = Path(output_dir) / expected_vis_path
            
            if full_vis_path.exists() and not force:
                logger.debug(f"Class3D visualization already exists: {expected_vis_path}")
                return job['name'], expected_vis_path
            
            # Generate new visualization
            vis_path = generate_class3d_visualization(
                job["details"].get("job_path", ""), 
                output_dir, 
                job['name']
            )
            return job['name'], vis_path
            
        except Exception as e:
            logger.error(f"Error in parallel processing of Class3D {job['name']}: {str(e)}")
            return job['name'], None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_class3d, job): job for job in class3d_jobs}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Class3D visualizations"):
            try:
                job_name, vis_path = future.result()
                if vis_path:
                    class3d_visualizations[job_name] = vis_path
            except Exception as e:
                job = futures[future]
                logger.error(f"Error processing Class3D job {job['name']}: {str(e)}")
    
    return class3d_visualizations

def get_job_note_filename(job):
    """Generate consistent filename for a job"""
    # Sanitize filename to avoid issues in different file systems
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", job['name'])
    return f"{safe_name}_{job['type']}.md"

def parse_refine3d_runout(job_dir):
    """Parse run.out file for Refine3D jobs to extract resolution and helical parameters"""
    result = {}
    run_out_path = Path(job_dir) / "run.out"
    
    if not run_out_path.exists():
        return result
    
    try:
        with open(run_out_path, 'r') as f:
            lines = f.readlines()
            
        # Search for resolution (Final resolution line)
        for line in lines:
            if "Final resolution (without masking) is:" in line:
                try:
                    resolution = line.split(":")[-1].strip()
                    result['resolution'] = float(resolution)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse resolution from line: {line}")
        
        # Search for helical parameters (use last occurrence)
        helical_lines = [line for line in lines if "Averaged helical twist" in line and "degrees, rise" in line and "Angstroms" in line]
        if helical_lines:
            last_helical_line = helical_lines[-1]
            try:
                # Parse: Averaged helical twist = -1.56311 degrees, rise = 4.79239 Angstroms.
                parts = last_helical_line.split("=")
                twist_part = parts[1].split("degrees")[0].strip()
                rise_part = parts[2].split("Angstroms")[0].strip()
                
                result['refined_twist'] = float(twist_part)
                result['refined_rise'] = float(rise_part)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse helical parameters from line: {last_helical_line}")
    
    except Exception as e:
        logger.error(f"Error parsing run.out for Refine3D: {str(e)}")
    
    return result

def parse_select_runout(job_dir):
    """Parse run.out file for Select jobs to extract particle or micrograph count"""
    run_out_path = Path(job_dir) / "run.out"
    
    if not run_out_path.exists():
        return {}
    
    result = {}
    try:
        with open(run_out_path, 'r') as f:
            for line in f:
                if "Written:" in line and "item(s)" in line:
                    try:
                        # Check if it's particles or micrographs
                        if "particles.star with" in line:
                            # Parse: Written: Select/job038/particles.star with 15366 item(s)
                            parts = line.split("with")
                            if len(parts) >= 2:
                                count_str = parts[1].split("item(s)")[0].strip()
                                result['particle_count'] = int(count_str)
                        elif "micrographs.star with" in line:
                            # Parse: Written: Select/job004/micrographs.star with 2543 item(s)
                            parts = line.split("with")
                            if len(parts) >= 2:
                                count_str = parts[1].split("item(s)")[0].strip()
                                result['micrograph_count'] = int(count_str)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse count from line: {line}")
    except Exception as e:
        logger.error(f"Error parsing run.out for Select: {str(e)}")
    
    return result

def parse_extract_runout(job_dir):
    """Parse run.out file for Extract jobs to extract particle count"""
    run_out_path = Path(job_dir) / "run.out"
    
    if not run_out_path.exists():
        return None
    
    try:
        with open(run_out_path, 'r') as f:
            for line in f:
                if "Written out STAR file with" in line and "particles in" in line:
                    try:
                        # Parse: Written out STAR file with 70138 particles in Extract/job018/particles.star
                        parts = line.split("with")
                        if len(parts) >= 2:
                            count_str = parts[1].split("particles in")[0].strip()
                            return int(count_str)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse particle count from line: {line}")
    except Exception as e:
        logger.error(f"Error parsing run.out for Extract: {str(e)}")
    
    return None

def parse_import_runout(job_dir):
    """Parse run.out file for Import jobs to extract item count"""
    run_out_path = Path(job_dir) / "run.out"
    
    if not run_out_path.exists():
        return None
    
    try:
        with open(run_out_path, 'r') as f:
            for line in f:
                if "Written" in line and "with" in line and "items" in line:
                    try:
                        # Parse: Written Import/job001/movies.star with 8200 items (8200 new items)
                        parts = line.split("with")
                        if len(parts) >= 2:
                            count_str = parts[1].split("items")[0].strip()
                            return int(count_str)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse item count from line: {line}")
    except Exception as e:
        logger.error(f"Error parsing run.out for Import: {str(e)}")
    
    return None

def copy_logfile_pdf(job_dir, job_name, output_dir):
    """Copy logfile.pdf to assets directory with job-specific name"""
    job_dir = Path(job_dir)
    if job_dir.is_file():
        job_dir = job_dir.parent
    
    logfile_path = job_dir / "logfile.pdf"
    
    if not logfile_path.exists():
        logger.warning(f"logfile.pdf not found in {job_dir}")
        return None
    
    try:
        # Sanitize job name for filename
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", job_name)
        dest_filename = f"{safe_name}_logfile.pdf"
        dest_path = Path(output_dir) / "assets" / dest_filename
        
        # Copy file
        shutil.copy2(logfile_path, dest_path)
        logger.info(f"Copied logfile.pdf to {dest_path}")
        
        return f"assets/{dest_filename}"
    except Exception as e:
        logger.error(f"Error copying logfile.pdf for {job_name}: {str(e)}")
        return None

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

def batch_generate_class2d_images(class2d_jobs, output_dir, force=False, max_workers=3):
    """
    Generate Class2D images in parallel using multiple threads.
    max_workers=3 is conservative to avoid memory issues with large images.
    """
    class2d_montages = {}
    
    def process_single_class2d(job):
        """Process a single Class2D job and return its montage path"""
        try:
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            job_nr = job_dir.name
            
            class2d_dir = job_dir
            model_star_files = list(class2d_dir.glob("*model.star"))
            
            if not model_star_files:
                logger.warning(f"No model.star files found for Class2D job {job_nr}")
                return job['name'], None
            
            model_star_files.sort()
            iteration = len(model_star_files) - 1
            expected_montage_path = f"assets/Class2D_{job_nr}_montage_It_{iteration}.png"
            full_image_path = Path(output_dir) / expected_montage_path
            
            if full_image_path.exists() and not force:
                logger.debug(f"Class2D image already exists: {expected_montage_path}")
                return job['name'], expected_montage_path
            
            # Generate new montage
            montage_path = generate_class2d_image(job["details"].get("job_path", ""), output_dir)
            return job['name'], montage_path
            
        except Exception as e:
            logger.error(f"Error in parallel processing of {job['name']}: {str(e)}")
            return job['name'], None
    
    # Process in parallel with limited workers to avoid memory issues
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(process_single_class2d, job): job for job in class2d_jobs}
        
        # Collect results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Class2D images"):
            try:
                job_name, montage_path = future.result()
                if montage_path:
                    class2d_montages[job_name] = montage_path
            except Exception as e:
                job = futures[future]
                logger.error(f"Error processing Class2D job {job['name']}: {str(e)}")
    
    return class2d_montages

def create_obsidian_notes(jobs, output_dir, force=False):
    """Create or update Obsidian notes for all jobs"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create assets directory for images
        assets_dir = os.path.join(output_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)
        
        # Load list of previously incomplete jobs
        incomplete_jobs = load_incomplete_jobs(output_dir)
        newly_incomplete_jobs = set()
        
        # Dictionary to store all jobs for quick lookups
        all_jobs = {}
        for job in jobs:
            all_jobs[job['name']] = job
        
        # Parse additional job-specific information only for jobs that need updates
        logger.info("Parsing job-specific information...")
        for job in tqdm(all_jobs.values(), desc="Parsing job details"):
            note_filename = get_job_note_filename(job)
            note_path = os.path.join(output_dir, note_filename)
            
            # Check if job was completed
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            
            success_file = job_dir / "RELION_JOB_EXIT_SUCCESS"
            job_is_complete = success_file.exists()
            
            # Determine if we should parse for this job
            should_parse = (
                force or  # Force flag is set
                not os.path.exists(note_path) or  # Note doesn't exist yet
                (job['name'] in incomplete_jobs and job_is_complete) or  # Was incomplete, now complete
                (job['name'] in incomplete_jobs and not job_is_complete)  # Still incomplete, update anyway
            )
            
            if not should_parse:
                continue
            
            # Refine3D specific parsing
            if job['type'] == 'Refine3D':
                refine_data = parse_refine3d_runout(job_dir)
                job['details'].update(refine_data)
            
            # Select specific parsing
            elif job['type'] == 'Select':
                select_data = parse_select_runout(job_dir)
                job['details'].update(select_data)
            
            # Extract specific parsing
            elif job['type'] == 'Extract':
                particle_count = parse_extract_runout(job_dir)
                if particle_count is not None:
                    job['details']['particle_count'] = particle_count
            
            # Import specific parsing
            elif job['type'] == 'Import':
                item_count = parse_import_runout(job_dir)
                if item_count is not None:
                    job['details']['item_count'] = item_count
            
            # Copy logfile.pdf for CtfFind and MotionCorr
            elif job['type'] in ['CtfFind', 'MotionCorr']:
                logfile_path = copy_logfile_pdf(job_dir, job['name'], output_dir)
                if logfile_path:
                    job['details']['logfile_pdf'] = logfile_path
        
        # Generate Class2D images before creating notes (parallel processing)
        logger.info("Checking Class2D visualizations...")
        class2d_jobs = [j for j in all_jobs.values() if j['type'] == 'Class2D']
        class2d_montages = batch_generate_class2d_images(class2d_jobs, output_dir, force, max_workers=3)

        # Generate Refine3D visualizations
        logger.info("Checking Refine3D visualizations...")
        refine3d_jobs = [j for j in all_jobs.values() if j['type'] == 'Refine3D']
        refine3d_visualizations = batch_generate_refine3d_visualizations(refine3d_jobs, output_dir, force, max_workers=2)
        
        # Generate Class3D visualizations
        logger.info("Checking Class3D visualizations...")
        class3d_jobs = [j for j in all_jobs.values() if j['type'] == 'Class3D']
        class3d_visualizations = batch_generate_class3d_visualizations(class3d_jobs, output_dir, force, max_workers=2)

        # First, create all notes to ensure they exist for cross-linking
        logger.info(f"Creating notes for {len(all_jobs)} jobs...")
        for job in tqdm(all_jobs.values(), desc="Creating notes"):
            note_filename = get_job_note_filename(job)
            note_path = os.path.join(output_dir, note_filename)
            
            # Check if job was completed
            job_dir = Path(job["details"].get("job_path", ""))
            if job_dir.is_file():
                job_dir = job_dir.parent
            
            success_file = job_dir / "RELION_JOB_EXIT_SUCCESS"
            job_is_complete = success_file.exists()
            
            # Determine if we should create/update the note
            should_update = (
                force or  # Force flag is set
                not os.path.exists(note_path) or  # Note doesn't exist yet
                (job['name'] in incomplete_jobs and job_is_complete) or  # Was incomplete, now complete
                (job['name'] in incomplete_jobs and not job_is_complete)  # Still incomplete, update anyway
            )
            
            if not should_update:
                continue
            
            # Track job completion status
            if not job_is_complete and job['type'] not in ['AutoPick']:
                newly_incomplete_jobs.add(job['name'])
                logger.debug(f"Job {job['name']} is incomplete - will be tracked for updates")
            elif job['name'] in incomplete_jobs:
                # Job is now complete, will be removed from tracking later
                logger.info(f"Job {job['name']} is now complete!")
                
            try:
                with open(note_path, "w", encoding='utf-8') as f:
                    # Add YAML frontmatter
                    f.write("---\n")
                    f.write(f"title: {job['name']} - {job['type']}\n")
                    f.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
                    
                    # Add tags from job details or defaults
                    tags = job['details'].get('tags', ["relion", job['type'].lower(), "cryo-em"])
                    # Add output folder name as tag
                    output_folder_name = os.path.basename(os.path.normpath(output_dir))
                    tags.append(output_folder_name)
                    # Convert tags list to string
                    tags_str = ', '.join([f'"{tag}"' for tag in tags])
                    f.write(f"tags: [{tags_str}]\n")
                    
                    # Add creation date from job details if available
                    creation_date = job['details'].get('creation_date', '')
                    if creation_date:
                        f.write(f"creation_date: {creation_date}\n")
                    
                    # Add job-specific properties to frontmatter
                    # Refine3D specific properties
                    if job['type'] == 'Refine3D':
                        if 'resolution' in job['details']:
                            f.write(f"resolution: {job['details']['resolution']:.2f}\n")
                        if 'refined_twist' in job['details']:
                            f.write(f"refined_twist: {job['details']['refined_twist']:.5f}\n")
                        if 'refined_rise' in job['details']:
                            f.write(f"refined_rise: {job['details']['refined_rise']:.5f}\n")
                    
                    # Select specific properties
                    if job['type'] == 'Select':
                        if 'particle_count' in job['details']:
                            f.write(f"particles: {job['details']['particle_count']}\n")
                        if 'micrograph_count' in job['details']:
                            f.write(f"micrographs: {job['details']['micrograph_count']}\n")
                    
                    # Extract specific properties
                    if job['type'] == 'Extract':
                        if 'particle_count' in job['details']:
                            f.write(f"particles: {job['details']['particle_count']}\n")
                    
                    # Import specific properties
                    if job['type'] == 'Import':
                        if 'item_count' in job['details']:
                            f.write(f"items: {job['details']['item_count']}\n")

                    f.write("---\n\n")
                    
                    # Job header
                    #f.write(f"# {job['name']}: {job['type']}\n\n")
                    
                    # Add Refine3D visualization if available
                    if job['type'] == 'Refine3D':
                        vis_path = refine3d_visualizations.get(job['name'])
                        if vis_path:
                            f.write(f"![Refine3D Analysis]({vis_path})\n\n")
                    
                    # Add Class3D visualization if available
                    if job['type'] == 'Class3D':
                        vis_path = class3d_visualizations.get(job['name'])
                        if vis_path:
                            f.write(f"![Class3D Analysis]({vis_path})\n\n")
                    
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
                            parts = input_job_name.replace(" ", "").rstrip("/").split("/")
                            if len(parts) == 2:
                                job_type, job_short = parts
                                link_basename = f"{job_short}_{job_type}"
                                link_text = f"{job_type}: {job_short}"
                                f.write(f"- [[{link_basename}|{link_text}]]\n")
                            else:
                                f.write(f"- {input_job_name}\n")
                        f.write("\n")


                    # Output jobs section (Jobs that use this)
                    output_jobs = job['details'].get('output_jobs', [])
                    if output_jobs:
                        f.write("## Jobs that use this\n\n")
                        for out_job_name in output_jobs:
                            # Beispiel: "ManualPick/job005"
                            # -> wir extrahieren type="ManualPick", short="job005"
                            parts = out_job_name.replace(" ", "").rstrip("/").split("/")
                            if len(parts) == 2:
                                job_type, job_short = parts
                                link_basename = f"{job_short}_{job_type}"
                                link_text = f"{job_type}: {job_short}"
                                f.write(f"- [[{link_basename}|{link_text}]]\n")
                            else:
                                f.write(f"- {out_job_name}\n")
                        f.write("\n")


                    
                    # # User notes section if available
                    # if 'user_notes' in job['details'] and job['details']['user_notes'].strip():
                    #     f.write("## User Notes\n\n")
                    #     f.write("```\n")
                    #     f.write(job['details']['user_notes'])
                    #     f.write("\n```\n\n")
                    
                    # # Details section
                    # f.write("## Additional Details\n\n")
                    # for key, value in job["details"].items():
                    #     if key not in ["settings", "input_jobs", "job_name", "job_path", "user_notes", "tags", "creation_date"]:
                    #         f.write(f"- **{key}**: {value}\n")
                    # f.write("\n")

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
                        f.write("| Setting | Value |\n")
                        f.write("|---------|-------|\n")
                        for key, value in job["details"]["settings"].items():
                            # Format value to handle long values
                            value_str = str(value)
                            if len(value_str) > 50:
                                value_str = value_str[:47] + "..."
                            f.write(f"| `{key}` | `{value_str}` |\n")

                    # Special sections for specific job types
                    if job['type'] == "Class2D":
                        f.write("## 2D Class Averages\n\n")
                        montage_path = class2d_montages.get(job['name'])
                        if montage_path:
                            f.write(f"![Class2D Montage]({montage_path})\n\n")
                        else:
                            f.write("*Class averages could not be generated*\n\n")
                    
                    elif job['type'] == "CtfFind":
                        if 'logfile_pdf' in job['details']:
                            f.write("## CTF Estimation Logfile\n\n")
                            f.write(f"![CTF Logfile]({job['details']['logfile_pdf']})\n\n")
                    
                    elif job['type'] == "MotionCorr":
                        if 'logfile_pdf' in job['details']:
                            f.write("## Motion Correction Logfile\n\n")
                            f.write(f"![Motion Correction Logfile]({job['details']['logfile_pdf']})\n\n")
                        
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
                        
                    # User notes section at the end if available
                    if 'user_notes' in job['details'] and job['details']['user_notes'].strip():
                        f.write("## RELION Command\n\n")
                        f.write("```\n")
                        f.write(job['details']['user_notes'])
                        f.write("\n```\n\n")
            
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
        
        # Update incomplete jobs tracking
        save_incomplete_jobs(output_dir, newly_incomplete_jobs)
        
        # Log summary
        completed_jobs = incomplete_jobs - newly_incomplete_jobs
        if completed_jobs:
            logger.info(f"Completed jobs since last run: {', '.join(completed_jobs)}")
        if newly_incomplete_jobs:
            logger.info(f"Currently tracking {len(newly_incomplete_jobs)} incomplete jobs")
        
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
                    
                    f.write(f"| {note_filename.replace('.md', '')} | {creation_date} | {input_job_count} |\n")
                
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

def get_incomplete_jobs_file(output_dir):
    """Get path to the hidden file that tracks incomplete jobs"""
    return os.path.join(output_dir, ".incomplete_jobs.txt")

def load_incomplete_jobs(output_dir):
    """Load list of incomplete job names from hidden file"""
    incomplete_file = get_incomplete_jobs_file(output_dir)
    if os.path.exists(incomplete_file):
        try:
            with open(incomplete_file, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            logger.error(f"Error reading incomplete jobs file: {str(e)}")
            return set()
    return set()

def save_incomplete_jobs(output_dir, incomplete_jobs):
    """Save list of incomplete job names to hidden file"""
    incomplete_file = get_incomplete_jobs_file(output_dir)
    try:
        with open(incomplete_file, 'w', encoding='utf-8') as f:
            for job_name in sorted(incomplete_jobs):
                f.write(f"{job_name}\n")
        logger.info(f"Saved {len(incomplete_jobs)} incomplete jobs to {incomplete_file}")
    except Exception as e:
        logger.error(f"Error saving incomplete jobs file: {str(e)}")

def add_incomplete_job(output_dir, job_name):
    """Add a job to the incomplete jobs list"""
    incomplete_jobs = load_incomplete_jobs(output_dir)
    incomplete_jobs.add(job_name)
    save_incomplete_jobs(output_dir, incomplete_jobs)

def remove_incomplete_job(output_dir, job_name):
    """Remove a job from the incomplete jobs list"""
    incomplete_jobs = load_incomplete_jobs(output_dir)
    if job_name in incomplete_jobs:
        incomplete_jobs.remove(job_name)
        save_incomplete_jobs(output_dir, incomplete_jobs)
        return True
    return False

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
