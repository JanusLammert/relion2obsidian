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

def parse_relion_jobs(project_dir):
    """
    Parses the RELION project directory to extract job information.
    Optimized to reduce RAM usage by streaming file processing.

    :param project_dir: Path to the RELION project directory.
    :return: Generator yielding job dictionaries.
    """
    for root, dirs, files in os.walk(project_dir):
        for file in files:
            # Process only relevant files
            if file.endswith("job.star") or file.endswith("job.json"):
                job_path = os.path.join(root, file)
                job_name = os.path.basename(root)

                # Determine job type based on folder name
                job_type = next((t for t in [
                    "Class2D", "Class3D", "Refine3D", "Extract", "CtfFind",
                    "ManualPick", "MaskCreate", "Micrographs", "Motion_Corr",
                    "Select", "Import", "PostProcess"
                ] if t in root), "Unknown")

                # Parse job information (JSON or STAR)
                if file.endswith(".json"):
                    with open(job_path, "r") as f:
                        job_details = json.load(f)
                else:
                    job_details = {"job_name": job_name, "job_path": os.path.abspath(job_path)}

                # Optionally parse note.txt
                note_path = os.path.join(root, "note.txt")
                if os.path.exists(note_path):
                    settings = {}
                    with open(note_path, "r") as note_file:
                        command_section = False
                        for line in note_file:
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

                yield {"name": job_name, "details": job_details, "type": job_type}

def normalize(img):
    min_val, max_val = np.min(img), np.max(img)
    return img if min_val == max_val else (img - min_val) / (max_val - min_val)

def add_text(gray_image_array, top_text="Top Text", bottom_text="Bottom Text"):
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

def make_montage(stk, ncols=10):
    nrows = int(np.ceil(stk.shape[0] / ncols))
    return montage(stk, grid_shape=(nrows, ncols), channel_axis=-1)

def generate_class2d_image(job_dir, out_dir, iteration=-1):
    try:
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt
        import mrcfile
        import starfile

        # Locate necessary files
        class2d_dir = Path(job_dir).parent
        job_nr = Path(job_dir).parent.name

        model_star_files = sorted(class2d_dir.glob("*model.star"))
        particle_star_files = sorted(class2d_dir.glob("*data.star"))
        mrc_stack_files = sorted(class2d_dir.glob("*classes.mrcs"))

        if not model_star_files or not particle_star_files or not mrc_stack_files:
            return None

        iteration = iteration if iteration >= 0 else min(
            len(model_star_files), len(particle_star_files), len(mrc_stack_files)
        ) - 1

        # Read only required columns to save memory
        model_data = starfile.read(model_star_files[iteration])
        particles_data = starfile.read(particle_star_files[iteration])
        with mrcfile.open(mrc_stack_files[iteration], permissive=True) as f:
            mrc_stk = f.data

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

        # Save the montage to a file
        output_path = Path(out_dir) / f"Class2D_{job_nr}_montage_It_{iteration}.png"
        plt.imsave(output_path, montage_img, cmap="gray")

        return str(output_path)

    except Exception as e:
        print(f"Error generating montage for {job_dir}: {e}")
        return None

def create_obsidian_notes(jobs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    processed_jobs = set(os.listdir(output_dir))

    highlighted_settings = [
        "o", "iter", "blush", "sym", "helical_twist_initial",
        "helical_rise_initial", "helical_symmetry_search", "solvent_mask"
    ]

    for job in tqdm(jobs, desc="Processing jobs"):
        note_filename = f"{job['type']}_{job['name']}.md"
        if note_filename in processed_jobs:
            continue

        note_path = os.path.join(output_dir, note_filename)
        with open(note_path, "w") as f:
            f.write(f"# {job['type']}_{job['name']}\n\n")
            f.write("## Job Details\n")
            for key, value in job["details"].items():
                if key != "settings":
                    f.write(f"- **{key}**: {value}\n")

            highlighted = {
                k: v for k, v in job["details"].get("settings", {}).items() if k in highlighted_settings
            }
            if highlighted:
                f.write("\n## Highlighted Settings\n")
                for key, value in highlighted.items():
                    f.write(f"- **{key}**: {value}\n")

            if "settings" in job["details"]:
                f.write("\n## Settings\n")
                f.write("| Setting | Value |\n")
                f.write("|---------|-------|\n")
                for key, value in job["details"]["settings"].items():
                    f.write(f"| {key} | {value} |\n")

            if job['type'] == "Class2D":
                f.write("\n## Montage 2D-Class-Averages\n")
                montage_path = generate_class2d_image(job["details"].get("job_path", ""), output_dir)
                if montage_path:
                    f.write(f"\n![Class2D Montage]({montage_path})\n")

def main():
    parser = argparse.ArgumentParser(description="Generate Obsidian notes from RELION project jobs.")
    parser.add_argument("-i", "--project_dir", required=True, help="Path to the RELION project directory.")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to the output directory for Obsidian notes.")

    args = parser.parse_args()

    jobs = parse_relion_jobs(args.project_dir)
    create_obsidian_notes(jobs, args.output_dir)

if __name__ == "__main__":
    main()
