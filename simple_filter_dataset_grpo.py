import torch
from torch.utils.data import Dataset
import random
import numpy as np
import re
import math
import logging
import nltk
import json
import os
from tqdm.auto import tqdm

# Set up logging
logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SVC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../SVC")

L_min, L_min_cache = 200, 450
L_max, L_max_cache = 1200, 1050

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

@torch.no_grad()
def pass_at_k_tensor(n, cs, k):
    """
    :param n: total number of samples
    :param cs: tensor of correct samples, shape [B, N]
    :param k: k in pass@$k$
    """
    c = torch.sum(cs, dim=-1)
    
    # Create a mask for cases where n - c < k
    # In these cases, pass@k is 1.0, because you are guaranteed to pick a correct sample.
    pass_at_k_is_one = (n - c) < k
    
    # For the other cases, calculate 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    # This is equivalent to 1 - (n-c)! * (n-k)! / (n! * (n-c-k)!)
    # We use log-gamma for numerical stability.
    log_numerator = torch.lgamma(n - c + 1) + torch.lgamma(n - k + 1)
    log_denominator = torch.lgamma(n + 1) + torch.lgamma(n - c - k + 1)
    
    # Calculate pass@k for all elements
    pass_at_k_values = 1.0 - torch.exp(log_numerator - log_denominator)
    
    # Apply the mask
    result = torch.where(pass_at_k_is_one, 1.0, pass_at_k_values)
    
    return result
    

class SceneObjectDataset(Dataset):
    """
    Dataset that generates synthetic 3D scenes with objects and creates
    instructions for identifying specific object categories.
    """
    SYSTEM_PROMPT: str = (
        "Respond in the following format, potraying \"Apeiria\":\n"
        "[APEIRIA THINKS]\n"
        "<... thinking predure ...>\n"
        "[APEIRIA SPEAKS]\n"
        "Apeiria <... responses ...>"
    )
    
    def __init__(self, tokenizer, num_samples=1000, min_objects=10, max_objects=15, seed=42):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: The tokenizer to use for apply chat template
            num_samples: Number of samples to generate
            min_objects: Minimum number of objects per scene
            max_objects: Maximum number of objects per scene
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.tokenizer = tokenizer
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Define object categories with their typical size ranges (width, height, depth)
        self.categories = {
            "table": {"min_size": [0.8, 0.6, 0.6], "max_size": [2.0, 1.0, 1.0]},
            "chair": {"min_size": [0.4, 0.8, 0.4], "max_size": [0.6, 1.2, 0.6]},
            "sofa": {"min_size": [1.5, 0.8, 0.8], "max_size": [2.5, 1.0, 1.0]},
            "bookshelf": {"min_size": [0.8, 1.5, 0.3], "max_size": [1.2, 2.0, 0.5]},
            "bed": {"min_size": [1.5, 0.5, 2.0], "max_size": [2.0, 0.7, 2.5]},
            "desk": {"min_size": [1.0, 0.7, 0.6], "max_size": [1.8, 0.8, 0.8]},
            "cabinet": {"min_size": [0.8, 1.0, 0.4], "max_size": [1.2, 1.8, 0.6]},
            "lamp": {"min_size": [0.3, 1.0, 0.3], "max_size": [0.5, 1.5, 0.5]},
            "plant": {"min_size": [0.3, 0.5, 0.3], "max_size": [0.6, 1.0, 0.6]},
            "rug": {"min_size": [1.5, 0.02, 1.0], "max_size": [3.0, 0.05, 2.0]}
        }
        
        # Generate all samples
        self.samples = self._generate_samples()
        
    def _generate_random_object(self, obj_id):
        """Generate a random object with a random category, position, and size."""
        category = random.choice(list(self.categories.keys()))
        
        # Generate random position within a room (assuming room size is 10x10x10)
        position = [
            round(random.uniform(0, 10), 2),  # x
            round(random.uniform(0, 10), 2),  # y
            round(random.uniform(0, 10), 2)   # z
        ]
        
        # Generate random size based on category constraints
        size = [
            round(random.uniform(self.categories[category]["min_size"][0], 
                                self.categories[category]["max_size"][0]), 2),  # width
            round(random.uniform(self.categories[category]["min_size"][1], 
                                self.categories[category]["max_size"][1]), 2),  # height
            round(random.uniform(self.categories[category]["min_size"][2], 
                                self.categories[category]["max_size"][2]), 2)   # depth
        ]
        
        return {
            "id": obj_id,
            "category": category,
            "position": position,
            "size": size
        }
    
    def _generate_scene(self):
        """Generate a random scene with objects."""
        num_objects = random.randint(self.min_objects, self.max_objects)
        objects = [self._generate_random_object(i) for i in range(num_objects)]
        return objects
    
    def _format_object_set(self, objects):
        """Format the object set as a string."""
        object_strings = []
        for obj in objects:
            pos = obj["position"]
            size = obj["size"]
            object_strings.append(
                f"Object {obj['id']}: Category: {obj['category']}, "
                f"Position: ({pos[0]}, {pos[1]}, {pos[2]}), "
                f"Size: {size[0]} x {size[1]} x {size[2]}"
            )
        return "\n".join(object_strings)
    
    def _generate_samples(self):
        """Generate all samples for the dataset."""
        samples = []
        
        for _ in range(self.num_samples):
            # Generate a random scene
            scene_objects = self._generate_scene()
            
            # Select a random category to query about
            query_category = random.choice(list(self.categories.keys()))
            
            # Format the object set
            object_set = self._format_object_set(scene_objects)
            
            # Create the input prompt
            prompt = (
                f"These are all objects in the scene: \n{object_set}\n"
                f"Think about the scene first. Identify all {query_category}s in the scene and provide their IDs, locations, and sizes."
                "In final answer, respond with \"Apeiria found...\" or \"didn't find any...\", and a list of Object <ID>: At (..., ..., ...), size: ... x ... x ..."
            )

            prompt = f"{prompt}"
            
            # Filter objects of the requested category
            category_objects = [obj for obj in scene_objects if obj["category"] == query_category]
            
            # Generate thinking trace
            thinking_trace = (
                f"I need to find all {query_category}s in the scene.\n"
                f"Looking through the object list for objects with category '{query_category}'..."
            )
            
            if category_objects:
                thinking_trace += f"\nFound {len(category_objects)} {query_category}(s):"
                for obj in category_objects:
                    thinking_trace += f"\n- Object {obj['id']} is a {query_category}"
            else:
                thinking_trace += f"\nI didn't find any {query_category}s in the scene."
            
            # Generate the expected response
            if category_objects:
                response_body = f"Apeiria found {len(category_objects)} {query_category}(s) in the scene:"
                for obj in category_objects:
                    pos = obj["position"]
                    size = obj["size"]
                    response_body += f"\nObject {obj['id']}: At ({pos[0]}, {pos[1]}, {pos[2]}), size: {size[0]} x {size[1]} x {size[2]}"
            else:
                response_body = f"Apeiria didn't find any {query_category}s in the scene."
            
            expected_response = f"[APEIRIA THINKS]\n{thinking_trace}\n[APEIRIA SPEAKS]\n{response_body}" + "<|im_end|>" + self.tokenizer.eos_token

            # apply tokenizer chat template
            prompt_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            # Store both the formatted response and the raw object data for reward calculation
            samples.append({
                "prompt": prompt,
                "answer": expected_response,
                "category": query_category,
                "objects": category_objects,  # Store the raw object data for reward calculation
                "task_type": "filter",
            })

        # log some samples
        logger.info(f"Sample prompt: {samples[0]['prompt']}")
        logger.info(f"Sample expected response: {samples[0]['answer']}")
        
        return samples
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        return self.samples[idx]

    def extract_answer_from_model_output(self, text):
        """
        Extract the answer from the model's output.
        This is compatible with the GRPO training code.
        """
        # Extract content after [APEIRIA SPEAKS]
        speaks_pattern = r'\[APEIRIA SPEAKS\](.*?)(?:\Z|$)'
        speaks_match = re.search(speaks_pattern, text, re.DOTALL)
        
        if speaks_match:
            return speaks_match.group(1).strip()
        return None

    def extract_answer_from_dataset(self, text):
        """
        Extract the expected answer from the dataset.
        This is compatible with the GRPO training code.
        """
        # Extract content after [APEIRIA SPEAKS]
        speaks_pattern = r'\[APEIRIA SPEAKS\](.*?)(?:\Z|$)'
        speaks_match = re.search(speaks_pattern, text, re.DOTALL)
        
        if speaks_match:
            return speaks_match.group(1).strip()
        return None


class SceneRelateDataset(SceneObjectDataset):
    """
    Dataset that generates synthetic 3D scenes with objects and creates
    instructions for identifying objects with specific spatial relationships.
    """
    
    def __init__(self, tokenizer, num_samples=1000, min_objects=10, max_objects=15, seed=42):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: The tokenizer to use for apply chat template
            num_samples: Number of samples to generate
            min_objects: Minimum number of objects per scene
            max_objects: Maximum number of objects per scene
            seed: Random seed for reproducibility
        """
        # Call the parent class constructor without generating samples yet
        super().__init__(tokenizer, 1, min_objects, max_objects, seed)
        
        # Set the number of samples for this dataset
        self.num_samples = num_samples
        
        # Define spatial relations
        self.relations = {
            "left": "to the left of",
            "right": "to the right of",
            "above": "above",
            "below": "below",
            "behind": "behind",
            "in_front": "in front of",
            "near": "near",
            "far": "far from"
        }
        
        # Generate samples for the relate task
        self.samples = self._generate_relate_samples()
    
    def _check_relation(self, obj_a, obj_b, relation):
        """
        Check if object A has the specified relation to object B.
        
        Args:
            obj_a: First object
            obj_b: Second object
            relation: Spatial relation to check
        
        Returns:
            bool: True if the relation holds, False otherwise
        """
        # Extract positions
        a_pos = obj_a["position"]
        b_pos = obj_b["position"]
        
        # Check specific relations
        if relation == "left":
            return a_pos[0] < b_pos[0]
        elif relation == "right":
            return a_pos[0] > b_pos[0]
        elif relation == "above":
            return a_pos[1] > b_pos[1]
        elif relation == "below":
            return a_pos[1] < b_pos[1]
        elif relation == "behind":
            return a_pos[2] < b_pos[2]
        elif relation == "in_front":
            return a_pos[2] > b_pos[2]
        elif relation == "near":
            # Calculate Euclidean distance between centers
            distance = math.sqrt(
                (a_pos[0] - b_pos[0])**2 + 
                (a_pos[1] - b_pos[1])**2 + 
                (a_pos[2] - b_pos[2])**2
            )
            # Define "near" as within 2 units
            return distance < 2.0
        elif relation == "far":
            # Calculate Euclidean distance between centers
            distance = math.sqrt(
                (a_pos[0] - b_pos[0])**2 + 
                (a_pos[1] - b_pos[1])**2 + 
                (a_pos[2] - b_pos[2])**2
            )
            # Define "far" as more than 5 units
            return distance > 5.0
        else:
            return False
    
    def _generate_relate_samples(self):
        """Generate samples for the 'relate' task."""
        samples = []
        
        num_attempts = 0
        max_attempts = self.num_samples * 30  # Limit the number of attempts to avoid infinite loops
        
        while len(samples) < self.num_samples and num_attempts < max_attempts:
            num_attempts += 1
            
            # Generate a random scene
            scene_objects = self._generate_scene()
            
            # Select random categories A and B, and a random relation
            categories = list(self.categories.keys())
            category_a = random.choice(categories)
            # Ensure different categories for A and B to avoid confusion
            category_b = random.choice([c for c in categories if c != category_a])
            relation = random.choice(list(self.relations.keys()))
            relation_text = self.relations[relation]
            
            # Find objects that match category A and category B
            objects_a = [obj for obj in scene_objects if obj["category"] == category_a]
            objects_b = [obj for obj in scene_objects if obj["category"] == category_b]
            
            # If either category has no objects, continue to next iteration
            if not objects_a or not objects_b:
                continue
            
            # Check which objects of category A have the specified relation to any object of category B
            valid_pairs = []
            for obj_a in objects_a:
                for obj_b in objects_b:
                    if obj_a["id"] != obj_b["id"] and self._check_relation(obj_a, obj_b, relation):
                        valid_pairs.append((obj_a, obj_b))
            
            # If no valid relations, continue to next iteration
            if not valid_pairs:
                continue
            
            # Select a random valid pair
            obj_a, obj_b = random.choice(valid_pairs)
            
            # Verify that obj_a is the only object of category_a with the relation to obj_b
            # (This ensures the task has a unique answer)
            other_valid = False
            for other_obj in objects_a:
                if other_obj["id"] != obj_a["id"] and self._check_relation(other_obj, obj_b, relation):
                    other_valid = True
                    break
            
            if other_valid:
                continue  # Skip this scene if there's more than one valid object
            
            # Format the object set
            object_set = self._format_object_set(scene_objects)
            
            # Create the input prompt
            prompt = (
                f"These are all objects in the scene: \n{object_set}\n"
                f"The order of axes is width (left-right) x height (low-high) x depth (forward-backward).\n"
                f"Think about the scene first. Identify the {category_a} that is {relation_text} "
                f"the {category_b}.\n"
                f"In your final answer, respond with \"Apeiria found...\" or \"didn't find any...\", "
                f"and provide in format Object <ID>: At (..., ..., ...), size: ... x ... x ..."
            )
            
            # Generate thinking trace
            thinking_trace = (
                f"I need to find the {category_a} that is {relation_text} the {category_b} (ID {obj_b['id']}).\n\n"
                f"First, let me locate the {category_b} with ID {obj_b['id']}.\n"
                f"The {category_b} (ID {obj_b['id']}) is at position ({obj_b['position'][0]}, {obj_b['position'][1]}, {obj_b['position'][2]}) "
                f"with size {obj_b['size'][0]} x {obj_b['size'][1]} x {obj_b['size'][2]}.\n\n"
                f"Now, I need to find a {category_a} that is {relation_text} this {category_b}.\n"
            )
            
            # Add reasoning about checking each object of category A
            for obj in objects_a:
                if obj["id"] == obj_a["id"]:  # This is the correct object
                    thinking_trace += (
                        f"Object {obj['id']} is a {category_a} at position ({obj['position'][0]}, {obj['position'][1]}, {obj['position'][2]}) "
                        f"with size {obj['size'][0]} x {obj['size'][1]} x {obj['size'][2]}.\n"
                        f"Checking if it is {relation_text} the {category_b} (ID {obj_b['id']})... Yes, it is.\n"
                    )
                else:
                    thinking_trace += (
                        f"Object {obj['id']} is a {category_a} at position ({obj['position'][0]}, {obj['position'][1]}, {obj['position'][2]}) "
                        f"with size {obj['size'][0]} x {obj['size'][1]} x {obj['size'][2]}.\n"
                        f"Checking if it is {relation_text} the {category_b} (ID {obj_b['id']})... No, it is not.\n"
                    )
            
            thinking_trace += f"\nTherefore, the {category_a} that is {relation_text} the {category_b} (ID {obj_b['id']}) is Object {obj_a['id']}."


            thinking_trace = "...thinking trace..."
            
            # Generate the expected response
            response_body = (
                f"Apeiria found the {category_a} that is {relation_text} the {category_b} (ID {obj_b['id']}):\n"
                f"Object {obj_a['id']}: At ({obj_a['position'][0]}, {obj_a['position'][1]}, {obj_a['position'][2]}), "
                f"size: {obj_a['size'][0]} x {obj_a['size'][1]} x {obj_a['size'][2]}"
            )
            
            expected_response = f"[APEIRIA THINKS]\n{thinking_trace}\n[APEIRIA SPEAKS]\n{response_body}" + "<|im_end|>" + self.tokenizer.eos_token

            # apply tokenizer chat template
            prompt_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            # Store the sample
            samples.append({
                "prompt": prompt,
                "answer": expected_response,
                "category_a": category_a,
                "category_b": category_b,
                "relation": relation,
                "object_a": obj_a,
                "object_b": obj_b,
                "task_type": "relate",
                "objects": [obj_a]  # For reward calculation, only the identified object matters
            })
        
        # Log some samples
        if samples:
            logger.info(f"Sample relate prompt: {samples[0]['prompt']}")
            logger.info(f"Sample relate expected response: {samples[0]['answer']}")

        # Log final number of attempts
        logger.info(f"Total attempts: {num_attempts}")
        logger.info(f"Total samples generated: {len(samples)}")
        
        return samples

class Sr3DReasoningDataset:
    """使用真实Sr3D数据的3D推理任务数据集。"""
    
    SYSTEM_PROMPT: str = (
        "Respond in the following format, potraying \"Apeiria\":\n"
        "[APEIRIA THINKS]\n"
        "<... thinking predure ...>\n"
        "[APEIRIA SPEAKS]\n"
        "Apeiria <... responses ...>"
    )
    
    def __init__(self, tokenizer, split="train", max_objects=80, seed=42):
        """初始化数据集"""
        self.tokenizer = tokenizer
        self.data_path = DATA_PATH
        self.split = split
        self.max_objects = max_objects
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 加载Sr3D注释数据
        self.annotation_file = self._get_annotation_file(split)
        self.annotations = self._load_annotations()
        
        # 加载场景数据
        self.scene_data = self._load_scene_data()
        
        # 生成样本
        self.samples = self._generate_samples_from_annotations()

        # self.stat_prompt_length() 
        # without trimming, min=424, max=4570, mean=1837, std=699, on val set
        # trim to max 80, min=424, max=3608, mean=1834, std=692, on val set
        
        # 记录示例样本
        if self.samples:
            logger.info(f"Sr3D样本提示: {self.samples[0]['prompt']}")
            logger.info(f"Sr3D预期响应: {self.samples[0]['answer']}")
    
    def _get_annotation_file(self, split):
        """获取注释文件路径"""
        SR3D_ANNO = {
            "train": f"{self.data_path}/sr3d_with_programs_train.json",
            "val": f"{self.data_path}/sr3d_with_programs_val.json",
        }
        return SR3D_ANNO[split]
    
    def _load_annotations(self):
        """从Sr3D文件加载注释"""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        logger.info(f"从{self.annotation_file}加载了{len(annotations)}条注释")
        return annotations
    
    def _pre_filter_objects(self, objects):
        BAD_OBJECTS = ["wall", "floor", "ceiling", "object"]
        return [obj for obj in objects if obj["name"] not in BAD_OBJECTS]
    
    def _load_scene_data(self):
        """加载所有场景的场景数据"""
        scene_data = {}
        scene_ids = set(anno["scene_id"] for anno in self.annotations)

        num_objects = []
        
        for scene_id in scene_ids:
            scene_file = f"{self.data_path}/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed/{scene_id}.json"
            try:
                with open(scene_file, 'r') as f:
                    scene_info = json.load(f)
                    # copy "locations" to "positions" for compatibility with the GRPO training code
                    for i, obj in enumerate(scene_info["objects"]):
                        scene_info["objects"][i]["position"] = obj["location"]
                    
                    # if self.pre_filter_objects:
                    scene_info["objects"] = self._pre_filter_objects(scene_info["objects"])[:self.max_objects]
                    num_objects.append(len(scene_info["objects"]))
                        
                    scene_data[scene_id] = scene_info
            except FileNotFoundError:
                logger.warning(f"场景文件未找到: {scene_file}")
        
        logger.info(f"加载了{len(scene_data)}个场景数据文件")
        # log some statistics of num_objects
        num_objects = np.array(num_objects)
        logger.info(f"场景对象数量: min={num_objects.min()}, max={num_objects.max()}, mean={num_objects.mean():.2f}, std={num_objects.std():.2f}, median={np.median(num_objects):.2f}")

        return scene_data
    
    def _format_object_set(self, objects):
        """将对象集格式化为字符串"""
        object_strings = []
        for obj in objects:
            # filter off wall,floor,ceiling
            if obj["name"] in ["wall", "floor", "ceiling"]:
                continue

            if len(object_strings) >= self.max_objects:
                break

            location = obj.get("location", [0, 0, 0])
            size = obj.get("size", [0, 0, 0])

            # times 100 and round to interger
            location = [round(coord * 100) for coord in location]
            size = [round(dim * 100) for dim in size]

            object_strings.append(
                f"Object {obj['id']}: {obj['name']} "
                f"at {location[0]}, {location[1]}, {location[2]}, "
                f"size {size[0]} x {size[1]} x {size[2]}"
            )
        return "\n".join(object_strings)
    
    def _generate_samples_from_annotations(self):
        """从Sr3D注释生成样本"""
        samples = []
        
        for anno in self.annotations:
            scene_id = anno["scene_id"]
            
            if scene_id not in self.scene_data:
                continue
                
            description = anno["description"]
            program = anno["program"]
            object_id = int(anno["object_id"]) if "object_id" in anno else None
            
            # 根据程序确定任务类型
            task_type = "filter"  # 默认
            if "relate(" in program:
                task_type = "relate"
            
            # 获取此场景的对象
            objects = self.scene_data[scene_id]["objects"]
            
            # 格式化对象集
            object_set = self._format_object_set(objects)
            
            # 创建输入提示
            prompt = (
                f"All objects: \n{object_set}\n"
                f"Find: \"{description}\"\n"
                f"In answer, respond with \"Apeiria found...\" or \"didn't find any...\", and a Object <ID>: At (..., ..., ...), size: ... x ... x ..."
            )
            
            # 根据object_id查找目标对象
            target_objects = []
            if object_id is not None:
                for obj in objects:
                    if obj["id"] == object_id:
                        target_objects.append(obj)

            if not target_objects:
                continue
            
            # 生成思考痕迹
            thinking_trace = (
                f"I need to find the object described as: \"{description}\".\n"
                f"The program for this task is: {program}\n"
            )
            
            if task_type == "filter":
                thinking_trace += "This task involves filtering objects by certain properties."
            elif task_type == "relate":
                thinking_trace += "This task involves finding objects with specific spatial relationships."
            
            if target_objects:
                thinking_trace += "\nTarget objects found:"
                for obj in target_objects:
                    location = obj.get("location", [0, 0, 0])
                    size = obj.get("size", [0, 0, 0])
                    thinking_trace += f"\n- Object {obj['id']} is a {obj['name']} at position ({location[0]}, {location[1]}, {location[2]}) with size {size[0]} x {size[1]} x {size[2]}"
            else:
                thinking_trace += f"\nI couldn't find any objects matching the description."
            
            # 生成预期的响应
            if target_objects:
                response_body = f"Apeiria found {len(target_objects)} object(s) matching the description:"
                for obj in target_objects:
                    location = obj.get("location", [0, 0, 0])
                    size = obj.get("size", [0, 0, 0])
                    response_body += f"\nObject {obj['id']}: At ({location[0]}, {location[1]}, {location[2]}), size: {size[0]} x {size[1]} x {size[2]}"
            else:
                response_body = f"Apeiria didn't find any objects matching the description."
            
            expected_response = f"[APEIRIA THINKS]\n{thinking_trace}\n[APEIRIA SPEAKS]\n{response_body}" + "<|im_end|>" + self.tokenizer.eos_token

            # 应用tokenizer聊天模板
            prompt_messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

            # 存储格式化的响应和原始对象数据
            samples.append({
                "prompt": formatted_prompt,
                "answer": expected_response,
                "description": description,
                "objects": target_objects,
                "task_type": task_type,
                "program": program,
                "scene_id": scene_id,
                "object_id": object_id,
            })
        
        return samples

    def stat_prompt_length(self):
        """统计提示长度"""
        # prompt_lengths = np.array([sample["prompt_len"] for sample in self.samples])
        prompt_lengths = []
        chunk_size = 100
        logger.info("计算提示长度...")
        for i in tqdm(range(0, len(self.samples), chunk_size)):
            chunk = self.samples[i:i + chunk_size]
            tokenized = self.tokenizer([sample["prompt"] for sample in chunk])
            prompt_lengths.extend([len(input_ids) for input_ids in tokenized["input_ids"]])
        prompt_lengths = np.array(prompt_lengths)
        logger.info(f"Prompt长度: min={prompt_lengths.min()}, max={prompt_lengths.max()}, mean={prompt_lengths.mean()}, std={prompt_lengths.std()}")


def parse_response(response: str):
    """
    Parse the model's response to extract object IDs and locations.
    
    Args:
        response: String response from the model
        
    Returns:
        List of dicts with keys: id, x, y, z, width, height, depth
    """
    parsed_objects = []

    # remove thinking trace if present, i.e., remove all contents before [APEIRIA SPEAKS]
    #  if not detected, but [APEIRIA THINKS] is detected, then the trace is truncated
    #  therefore, the response is invalid, return empty list
    # if re.search(r"\[APEIRIA THINKS\]", response, flags=re.IGNORECASE) is not None:
    #     if re.search(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE) is None:
    #         logger.warning("Thinking trace detected without response, skipping...")
    #         return parsed_objects

    #     else:
    #         # remove all contents before [APEIRIA SPEAKS]
    #         response = re.split(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE)[-1].strip()
    thinking, response = extract_thinking_and_answer(response)
    if response is None:
        return parsed_objects
    
    # Regular expressions for different response formats
    patterns = [
        # Pattern 1: Object X: At (x, y, z), size: w x h x d
        r"Object\s+(\d+):\s+At\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*size:\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
        
        # Pattern 2: ID X: Position (x, y, z), size w x h x d
        r"ID\s+(\d+):\s+Position\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*size\s+([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
        
        # Pattern 3: X: Coordinates (x, y, z), dimensions w x h x d
        r"(\d+):\s+(?:Coordinates\s+)?\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*(?:dimensions|size)?\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)",
        
        # Pattern 4: Object X: (x, y, z), w x h x d
        r"(?:Object\s+)?(\d+):\s+\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\),\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)\s*x\s*([+-]?\d+\.?\d*)"
    ]
    
    # Try each pattern
    for pattern in patterns:
        # let it be case-insensitive
        matches = re.finditer(pattern, response, re.IGNORECASE)
        for match in matches:
            try:
                obj_id = int(match.group(1))
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                width = float(match.group(5))
                height = float(match.group(6))
                depth = float(match.group(7))
                
                parsed_objects.append({
                    "id": obj_id,
                    "x": x, "y": y, "z": z,
                    "width": width, "height": height, "depth": depth
                })
            except (ValueError, IndexError):
                continue
    
    return parsed_objects


def calculate_position_similarity(pred_pos, true_pos):
    """
    Calculate similarity between predicted and true positions.
    Returns a value between 0 and 1, where 1 means perfect match.
    
    Args:
        pred_pos: [x, y, z] predicted position
        true_pos: [x, y, z] true position
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Calculate Euclidean distance
    distance = math.sqrt(
        (pred_pos[0] - true_pos[0])**2 + 
        (pred_pos[1] - true_pos[1])**2 + 
        (pred_pos[2] - true_pos[2])**2
    )
    
    # Convert distance to similarity score (1 for perfect match, decreasing as distance increases)
    # Using a sigmoid-like function that gives 0.5 at distance=1.0
    # similarity = 1.0 / (1.0 + distance)
    similarity = math.exp(-2 * distance)  # Exponential decay
    
    return similarity


def calculate_size_similarity(pred_size, true_size):
    """
    Calculate similarity between predicted and true sizes.
    Returns a value between 0 and 1, where 1 means perfect match.
    
    Args:
        pred_size: [width, height, depth] predicted size
        true_size: [width, height, depth] true size
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Calculate relative differences for each dimension
    width_diff = abs(pred_size[0] - true_size[0]) / max(true_size[0], 0.01)
    height_diff = abs(pred_size[1] - true_size[1]) / max(true_size[1], 0.01)
    depth_diff = abs(pred_size[2] - true_size[2]) / max(true_size[2], 0.01)
    
    # Average the differences and convert to similarity
    avg_diff = (width_diff + height_diff + depth_diff) / 3.0
    # similarity = 1.0 / (1.0 + 2.0 * avg_diff)  # Gives 0.5 at 50% average difference
    similarity = math.exp(-2 * avg_diff)  # Exponential decay
    
    return similarity

def extract_thinking_and_answer(response: str) -> tuple[str | None, str | None]:
    """
    Extract the thinking trace and the answer from the model's response.
    
    Args:
        response: Full response from the model
    """
    # 提取思考痕迹
    thinking_match = re.search(r'\[APEIRIA THINKS\](.*?)(?=\[APEIRIA SPEAKS\])', response, re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else None
    
    # 提取回答
    answer_match = re.search(r'\[APEIRIA SPEAKS\](.*)', response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None
    
    return thinking, answer
    

def fine_grained_reward(prompts, completions, answer_data, **kwargs):
    """
    Calculate a fine-grained reward based on object identification accuracy.
    
    Args:
        prompts: List of prompt texts
        completions: List of completion dictionaries
        answer_data: List of dictionaries containing ground truth data
        
    Returns:
        list: List of rewards for each completion
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    # reward_details_list = []
    
    for response, data in zip(responses, answer_data):
        # reward_details = {}

        # Extract ground truth objects
        true_objects = data.get("objects", [])
        # Parse the model's response to extract predicted objects
        thinking, answer = extract_thinking_and_answer(response)
        if answer is None:
            answer = "<no answer is parsed>"

        predicted_objects = parse_response(response)
        
        # Get task type
        task_type = data.get("task_type", "filter")
        
        # If there are no objects to find, check if model correctly reports this
        if not true_objects:
            if "didn't find any" in answer.lower() or "no " + data.get("category", "") in answer.lower():
                reward = 2.5  # Full reward for correctly reporting no objects
                # Still, penalize if there are objects predicted
                if predicted_objects:
                    reward -= 1.0 * len(predicted_objects)
                rewards.append(min(max(reward, 0.0), 2.5))
                # reward_details_list.append(reward_details)
            elif len(predicted_objects) == 0:
                rewards.append(2.5)  # Full reward for correctly reporting no objects
                logger.warning(f"Model correctly reported no objects without explicit denial phrase. Response: {response}")
                # reward_details_list.append(reward_details)
            else:
                rewards.append(0.0)  # No reward for incorrectly reporting objects
                # reward_details_list.append(reward_details)
            continue
        
        if not predicted_objects:
            rewards.append(0.0)  # No reward if no objects were parsed
            # reward_details_list.append(reward_details)
            continue
        
        # Convert true objects to the same format as predicted objects
        formatted_true_objects = []
        for obj in true_objects:
            formatted_true_objects.append({
                "id": obj["id"],
                "x": obj["position"][0],
                "y": obj["position"][1],
                "z": obj["position"][2],
                "width": obj["size"][0],
                "height": obj["size"][1],
                "depth": obj["size"][2]
            })
        
        # Create dictionaries for quick lookup
        true_obj_dict = {obj["id"]: obj for obj in formatted_true_objects}
        pred_obj_dict = {obj["id"]: obj for obj in predicted_objects}
        
        # For relate task, there should be exactly one object
        # if task_type == "relate" and len(predicted_objects) != 1:
        #     rewards.append(0.0)  # Incorrect number of objects
        #     continue
        
        # 1. Reward for correctly identifying object IDs (0.5 points per correct ID, devided by total true objects)
        correct_ids = set(true_obj_dict.keys()) & set(pred_obj_dict.keys())
        id_reward = len(correct_ids) * 0.5 # the division will be done later for filter task

        # 1.5. add soft reward that: if predicted more or equal to true, give discounted reward (depending on the additional predicted objects)
        MAX_REWARDING_EXTRA_ID_COUNT = 2 # FIXME: correct_ids here can never be larger than true_obj_dict.keys()
        if len(correct_ids) >= len(true_obj_dict): # if covered all true objects
            extra_ids = set(pred_obj_dict.keys()) - set(true_obj_dict.keys())
            extra_reward = 1 - len(extra_ids) / MAX_REWARDING_EXTRA_ID_COUNT
            extra_reward = max(-0.5, extra_reward) * 0.5
            id_reward += extra_reward
        
        # 2. Additional reward for position and size accuracy. (Max 1.5=0.75+0.75 points for each object)
        position_size_reward = 0.0
        for obj_id in correct_ids:
            true_obj = true_obj_dict[obj_id]
            pred_obj = pred_obj_dict[obj_id]
            
            # Position similarity (up to 0.75 points)
            pos_sim = calculate_position_similarity(
                [pred_obj["x"], pred_obj["y"], pred_obj["z"]],
                [true_obj["x"], true_obj["y"], true_obj["z"]]
            )
            # position_reward = 0.75 * (1 if pos_sim > 0.95 else 0)
            position_reward = 0.75 * pos_sim
            
            # Size similarity (up to 0.75 points)
            size_sim = calculate_size_similarity(
                [pred_obj["width"], pred_obj["height"], pred_obj["depth"]],
                [true_obj["width"], true_obj["height"], true_obj["depth"]]
            )
            # size_reward = 0.75 * (1 if size_sim > 0.95 else 0)
            size_reward = 0.75 * size_sim
            
            # Add to total position/size reward
            position_size_reward += position_reward + size_reward
        
        # Calculate total reward
        total_reward = id_reward + position_size_reward
        
        # For filter task, normalize by number of true objects
        if task_type == "filter":
            total_reward /= len(true_objects)

        # reward_details['id_reward'] = id_reward
        # reward_details['position_size_reward'] = position_size_reward
        
        # # -1. Length Shaping: penalize very short answers and very long answers
        # penalties = length_penalty(
        #     responses=[response],
        #     # L_min=200, L_min_cache=450,
        #     # L_max=1200, L_max_cache=1050,
        #     L_min=L_min, L_min_cache=L_min_cache,
        #     L_max=L_max, L_max_cache=L_max_cache
        # )
        # total_reward += penalties[0]  # there is only one completion here

        # reward_details['length_penalty'] = penalties[0]

        # Ensure reward is non-negative and cap at maximum
        # total_reward = max(0.0, min(total_reward, 2.5))
        
        rewards.append(total_reward)
        # reward_details_list.append(reward_details)
    
    return rewards # , reward_details_list

def length_penalty(responses, L_min, L_min_cache, L_max, L_max_cache):
    """
    Apply length penalty to the responses.
    
    Args:
        responses: List of model responses
        L_min: Minimum length threshold
        L_min_cache: Cache for minimum length
        L_max: Maximum length threshold
        L_max_cache: Cache for maximum length

    Returns:
        list: List of length penalties
    """
    penalties = []
    for response in responses:
        # use nltk to tokenize the content
        tokens = nltk.word_tokenize(response)
        length = len(tokens)
        
        if length < L_min or length > L_max:
            penalty = -1.0
        elif L_min_cache <= length <= L_max_cache:
            penalty = 0.0
        elif L_min <= length < L_min_cache:
            # shaping that, linearly increase from -1.0 to 0.0, from L_min to L_min_cache
            penalty = - (L_min_cache - length) / (L_min_cache - L_min)
        elif L_max_cache < length <= L_max:
            penalty = - (length - L_max_cache) / (L_max - L_max_cache)

        penalties.append(penalty)
    return penalties

def length_reward(prompts, completions, answer_data, **kwargs):
    """
    Assigns a reward based on the length of the completion.
    
    Args:
        prompts: List of prompt texts
        completions: List of completion dictionaries
        answer_data: List of dictionaries containing ground truth data

    Returns:
        list: List of length-based rewards
    """
    # slowly increase the reward of length upto 1000 characters
    rewards = []
    for completion in completions:
        content = completion[0]['content']
        # use nltk to tokenize the content
        tokens = nltk.word_tokenize(content)
        reward = min(1.0, len(tokens) / 1000)
        rewards.append(reward)
    return rewards

def format_reward(completions, answer_data, **kwargs):
    """
    Assigns a reward for adhering to the desired format.
    
    Args:
        completions: List of model completions
        
    Returns:
        list: List of format compliance scores
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    
    for response, data in zip(responses, answer_data):
        thinking_trace, answer = extract_thinking_and_answer(response)

        score = 0.0
        if "[APEIRIA THINKS]\n" in response: score += 0.1
        if "[APEIRIA SPEAKS]\n" in response: score += 0.1

        # make sure [APEIRIA SPEAKS] is after [APEIRIA THINKS], note that it shall detect across newlines
        if re.search(r"\[APEIRIA THINKS\].*\[APEIRIA SPEAKS\]", response, re.DOTALL):
            score += 0.1
        
        # if no gt objects, then the response should contain "didn't find any"
        not_found = re.search(r"Apeiria didn't find any", response)
        do_found = re.search(r"Apeiria found", response)
        if answer is not None:
            has_object_info = re.search(r"Object\s+\d+:\s+At\s+\([^)]+\),\s+size:", answer)
        else:
            has_object_info = False

        if not data.get("objects", []): # if don't have target objects
            if not_found and not do_found:
                score += 0.1
            if not has_object_info:
                score += 0.1
        else:
            if not not_found and do_found:
                score += 0.1

            if has_object_info:
                score += 0.1

        # Ensure there is "plan stage"
        # the keywords: "Let's plan my next steps"
        # if no, penalize 0.1
        if thinking_trace is not None:
            if re.search(r"let's plan", thinking_trace, re.IGNORECASE):
                pass
            else:
                score -= 0.1


        # Put length penalty here as well
        penalties = length_penalty(
            responses=[response],
            L_min=L_min, L_min_cache=L_min_cache,
            L_max=L_max, L_max_cache=L_max_cache
        )
        score += penalties[0]

        # max 0.5
        rewards.append(score)
    
    return rewards


def combined_reward(prompts, completions, answer_data):
    """
    Combines fine-grained correctness and format rewards.
    
    Args:
        prompts: List of prompt texts
        completions: List of completion dictionaries
        answer_data: List of dictionaries containing ground truth data
        
    Returns:
        list: Combined rewards
    """
    # Get individual rewards
    correctness_scores = fine_grained_reward(prompts=prompts, completions=completions, answer_data=answer_data)
    format_scores = format_reward(completions=completions, answer_data=answer_data)
    
    # Combine rewards
    combined_rewards = []
    reward_info = []
    for c_score, f_score in zip(correctness_scores, format_scores):
        # Correctness score range: 0.0 to 2.0
        # Format score range: 0.0 to 0.8
        # Total range: 0.0 to 2.8
        combined_rewards.append(c_score + f_score)
        reward_info.append({"correctness_score": c_score, "format_score": f_score})
    
    return combined_rewards, reward_info

class CombinedReward:
    def __init__(self, do_max_reward_normalize=True, logp_factor_correct=0.0, logp_factor_wrong=0.0, **kwargs):
        self.do_max_reward_normalize = do_max_reward_normalize
        self.logp_factor_correct = logp_factor_correct
        self.logp_factor_wrong = logp_factor_wrong
        
    def __call__(self, prompts, completions, answer_data, **kwargs):
        rewards, reward_info = combined_reward(prompts, completions, answer_data)
        if self.do_max_reward_normalize:
            rewards = [min(r, self.max_reward) for r in rewards]
        return rewards

    def compute_correctness(self, prompts, completions, answer_data):
        """
        returns a list of accuracy values (0 or 1) based on correctness_score
        1 if correctness_score >= reward_considered_correct else 0
        """
        rewards, reward_info = combined_reward(prompts, completions, answer_data)
        return [1 if info["correctness_score"] >= self.reward_considered_correct else 0 for info in reward_info]

    def calibrate_rewards(self, rewards, log_probs):
        """
        Calibrate rewards by adjusting with log probabilities.
        
        Args:
            rewards: List of rewards
            log_probs: List of log probabilities for each completion
            
        Returns:
            list: Calibrated rewards
        """
        calibrated_rewards = []
        for r, logp in zip(rewards, log_probs):
            if r >= self.reward_considered_correct:
                r -= self.logp_factor_correct * logp # lopg is negative, so this is a reward to less likely output (encourage exploration)
            else:
                r += self.logp_factor_wrong * logp # logp is negative, so this is a penalty to more likely output
            calibrated_rewards.append(r)
        
        return calibrated_rewards

    @property
    def max_reward(self):
        return 2.8

    @property
    def reward_considered_correct(self):
        return 2.2


# Prepare dataset for GRPO training
def prepare_scene_dataset(tokenizer, num_samples=1000, eval_size=50):
    """
    Prepare the scene dataset for GRPO training.
    
    Args:
        num_samples: Number of samples to generate
        eval_size: Number of samples to use for evaluation
    
    Returns:
        train_data, eval_data: Training and evaluation datasets
    """
    # Create the dataset
    dataset = SceneObjectDataset(tokenizer, num_samples=num_samples)
    
    # Split into training and evaluation sets
    all_data = dataset.samples
    random.shuffle(all_data)
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    
    return train_data, eval_data

# Prepare dataset for GRPO training
def prepare_scene_dataset_relates(tokenizer, num_samples=1000, eval_size=50):
    """
    Prepare the scene dataset for GRPO training.
    
    Args:
        num_samples: Number of samples to generate
        eval_size: Number of samples to use for evaluation
    
    Returns:
        train_data, eval_data: Training and evaluation datasets
    """
    # Create the dataset
    dataset = SceneRelateDataset(tokenizer, num_samples=num_samples)
    
    # Split into training and evaluation sets
    all_data = dataset.samples
    random.shuffle(all_data)
    eval_data = all_data[:eval_size]
    train_data = all_data[eval_size:]
    
    return train_data, eval_data

def prepare_sr3d_dataset(tokenizer, eval_ratio=0.1):
    """
    准备Sr3D数据集用于GRPO训练
    
    Args:
        tokenizer: 要使用的tokenizer
        eval_ratio: 要使用的数据比例（快速测试用）
    
    Returns:
        train_data, eval_data: 训练和评估数据集
    """
    # 创建数据集
    dataset = Sr3DReasoningDataset(tokenizer, split="train")
    val_dataset = Sr3DReasoningDataset(tokenizer, split="val")
    
    # 取数据子集进行测试
    train_data = dataset.samples
    eval_data = val_dataset.samples
    eval_data = random.sample(eval_data, int(len(eval_data) * eval_ratio))
    
    logger.info(f"准备了{len(train_data)}个训练样本和{len(eval_data)}个评估样本")
    
    return train_data, eval_data

# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create a small dataset
    dataset = SceneObjectDataset(num_samples=5)
    
    # Display a sample
    sample = dataset[0]
    print("Sample prompt:")
    print(sample["prompt"])
    print("\nSample expected response:")
    print(sample["answer"])
    
    # Test the reward function
    test_completion = [[{"content": sample["answer"]}]]  # Perfect answer
    test_reward = fine_grained_reward(
        prompts=[sample["prompt"]], 
        completions=test_completion, 
        answer_data=[sample]
    )
    print(f"\nReward for perfect answer: {test_reward[0]}")
    
    # Test with a partially correct answer
    # Modify one object's position slightly
    modified_answer = sample["answer"].replace(
        "At (3.75, 9.51, 7.32)", 
        "At (3.85, 9.61, 7.42)"
    )
    test_completion = [[{"content": modified_answer}]]
    test_reward = fine_grained_reward(
        prompts=[sample["prompt"]], 
        completions=test_completion, 
        answer_data=[sample]
    )
    print(f"Reward for slightly modified answer: {test_reward[0]}")
    
    # Test with a missing object
    lines = sample["answer"].split("\n")
    modified_lines = [line for line in lines if "Object 10:" not in line]
    modified_answer = "\n".join(modified_lines)
    test_completion = [[{"content": modified_answer}]]
    test_reward = fine_grained_reward(
        prompts=[sample["prompt"]], 
        completions=test_completion, 
        answer_data=[sample]
    )
    print(f"Reward for answer with missing object: {test_reward[0]}")
