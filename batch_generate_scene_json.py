import os
import subprocess
import glob
from tqdm.auto import tqdm
import multiprocessing
from functools import partial

PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
scannet_dir = "/network_space/server129/shared_dataset/scannet/scans"
scannet_test_dir = "/network_space/server129/shared_dataset/scannet/scans_test"
output_dir = os.path.join(PROJECT_HOME, "scannet/scans_fixed")
label_map_file = os.path.join(PROJECT_HOME, "data/scannetv2-labels.combined.tsv")
# data_dir = os.path.join(PROJECT_HOME, "data/apeiria_scannet")
# data_dir = os.path.join(PROJECT_HOME, "data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed") # used to save 3D scene json for apeiria
data_dir = os.path.join(PROJECT_HOME, "data/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4") # used to save 3D scene json for apeiria


os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

scene_dirs = glob.glob(os.path.join(scannet_dir, "scene*_*")) + glob.glob(os.path.join(scannet_test_dir, "scene*_*"))

scene_dirs = [d for d in glob.glob(os.path.join(scannet_dir, "scene*_*")) if os.path.isdir(d)]


print(f"Found {len(scene_dirs)} scenes to process.")

def process_scene(scene_dir, output_dir, label_map_file, data_dir):
    scene_name = os.path.basename(scene_dir)
    
    # Step 1: Generate bbox processed file
    # cmd1 = [
    #     "python", 
    #     "load_scannet_detailed_scene.py",
    #     "--scan_path", scene_dir,
    #     "--output_file", os.path.join(output_dir, scene_name),
    #     "--label_map_file", label_map_file
    # ]
    
    # subprocess.run(cmd1, cwd="./scannet", check=True)

    # Step 2: Generate scene json representation
    input_file = os.path.join(output_dir, f"{scene_name}_aligned_bbox.npy")
    output_file = os.path.join(data_dir, f"{scene_name}.json")
    
    cmd2 = [
        "python",
        "generate_scene_json_from_bbox_list.py",
        "-i", input_file,
        "-o", output_file,
        # "-c", "data/scene_object_top_captions_by_itc.json"
        "-c", "data/scene_object_top_captions_from_gpt4o.json",
        "-p", "4"
    ]
    
    subprocess.run(cmd2, check=True)
    
    return scene_name

if __name__ == '__main__':
    # num_processes = multiprocessing.cpu_count()
    num_processes = 12
    print(f"Using {num_processes} processes")

    pool = multiprocessing.Pool(processes=num_processes)
    
    process_func = partial(process_scene, output_dir=output_dir, label_map_file=label_map_file, data_dir=data_dir)
    
    with tqdm(total=len(scene_dirs), desc="Processing scenes") as pbar:
        for _ in pool.imap_unordered(process_func, scene_dirs):
            pbar.update()

    pool.close()
    pool.join()

    print("All scenes processed.")
