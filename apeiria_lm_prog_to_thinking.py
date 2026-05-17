from lark import Lark, Transformer, v_args
import random
import numpy as np
import json
import re
import os
import logging
import math
from typing import Dict, List, Tuple, Set, Optional, Any, Union, Callable, Iterable
from enum import Enum
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch
from traceback import print_exc
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial
import nltk
import copy
from icecream import ic
import transformers
from filelock import FileLock
from contextlib import contextmanager
import lark
import math
from transforms3d.euler import quat2euler
import torch.distributed as dist
from functools import lru_cache
import lark_cython
from scipy.optimize import linear_sum_assignment

from eval_utils import score_captions

from apeiria_lm_utils import (
    Synthetic3DDataset, Synthetic3DRelationalDataset, Synthetic3DObjectInfoDataset, 
    ScanNetRawObject, box3d_iou_orthogonal, mutual_iou, mutual_iou_vectorized
)
from image_feature_manager import ImageFeatureManager

from qwen_helpers import apply_qwen_template, apply_qwen_template_with_partial_response

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SVC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../SVC")

random.seed(42)
np.random.seed(42)

def worker_init(shared_scene_data):
    global _worker_scene_data
    _worker_scene_data = shared_scene_data

@contextmanager
def locked_file(filename, mode='a'):
    lock = FileLock(f"{filename}.lock")
    with lock:
        with open(filename, mode) as f:
            yield f

# Spatial relation enumeration class
class SpatialRelation(Enum):
    LEFT = "left"
    RIGHT = "right"
    ABOVE = "above"
    BELOW = "below"
    BEHIND = "behind"
    IN_FRONT = "in front of"
    NEAR = "near"
    FAR = "far from"
    NEXT_TO = "next to"
    BESIDE = "beside"
    ON = "on"
    UNDER = "under"
    NONE = "no relation"

# Define program grammar
program_grammar = r"""
    program: function
    
    function: NAME "(" [argument ("," argument)*] ")"
    
    argument: function
            | STRING
            | NAME
    
    STRING: /"[^"]*"/
    NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    
    %import common.WS
    %ignore WS
"""

_GLOBAL_LARK_PARSER = Lark(program_grammar, start="program", parser="lalr", cache=True)


class ProgramTransformer(Transformer):
    def __init__(self, scene_data):
        super().__init__()
        self.scene_data = scene_data
        
    def program(self, items):
        return items[0]
    
    def function(self, items):
        # print(items)
        func_name = items[0]["value"]
        args = items[1:]
        return {"type": "function", "name": func_name, "args": args}
    
    def argument(self, items):
        return items[0]
    
    def NAME(self, token):
        return {"type": "name", "value": token.value}
    
    def STRING(self, token):
        # Remove quotes
        value = token.value[1:-1]
        return {"type": "string", "value": value}


# LRU-cached parse+transform (per process)
@lru_cache(maxsize=10_000)
def _parse_program_cached(program_str: str):
    tree = _GLOBAL_LARK_PARSER.parse(program_str)
    # Transformer is stateless for our grammar; reuse a single instance
    return ProgramTransformer(None).transform(tree)

def parse_program(program_str, scene_data):
    """Parse program string into AST (fast, cached, global parser)."""
    return _parse_program_cached(program_str)

# def parse_program(program_str, scene_data):
#     """Parse program string into AST"""
#     # parser = Lark(program_grammar, start="program", parser="lalr")
#     # tree = parser.parse(program_str)
#     tree = _GLOBAL_LARK_PARSER.parse(program_str)
#     return ProgramTransformer(scene_data).transform(tree)

def execute_program(ast_node, scene_data, target_object_id=None, execution_trace=None, dummy_execution=False):
    """Execute program AST and return results, categories, LABELS, and trace."""
    if execution_trace is None:
        execution_trace = []
    
    if ast_node["type"] == "function":
        func_name = ast_node["name"]
        arg_results = []
        arg_categories = []
        arg_labels = []  # <--- New: Track descriptions of arguments
        arg_traces = []
        
        # Execute all arguments first
        for arg in ast_node["args"]:
            if isinstance(arg, dict) and arg["type"] == "function":
                # Recursive call now returns label as well
                arg_result, arg_category, arg_label, arg_trace = execute_program(
                    arg, scene_data, target_object_id, [], dummy_execution=dummy_execution
                )
                arg_results.append(arg_result)
                arg_categories.append(arg_category)
                arg_labels.append(arg_label)
                arg_traces.extend(arg_trace)
            else:
                # Simple argument (name, string)
                if arg is not None:
                    val = arg["value"]
                    arg_results.append(val)
                    arg_categories.append(None)
                    arg_labels.append(str(val)) # Simple args describe themselves
                else:
                    arg_results.append(None)
                    arg_categories.append(None)
                    arg_labels.append("unknown")
        
        # Add all argument execution traces to main trace
        execution_trace.extend(arg_traces)
        
        # Execute the function itself
        # Now returns a label as well
        result, category, label = execute_function(
            func_name, arg_results, arg_categories, arg_labels, 
            scene_data, target_object_id, dummy_execution=dummy_execution
        )
        
        # Add this function execution to trace
        trace_entry = {
            "function": func_name,
            "arguments": arg_results,
            "arg_labels": arg_labels, # <--- Store arg labels for trace generation
            "result": result,
            "result_category": category,
            "result_label": label,    # <--- Store result label
            "arg_categories": arg_categories,
        }
        execution_trace.append(trace_entry)
        
        return result, category, label, execution_trace
    
    elif ast_node["type"] == "name":
        return ast_node["value"], None, ast_node["value"], execution_trace
    
    elif ast_node["type"] == "string":
        return ast_node["value"], None, ast_node["value"], execution_trace

def execute_function(func_name, args, arg_categories, arg_labels, scene_data, target_object_id=None, dummy_execution=False):    
    """Execute specific function with its arguments"""
    # --- 1. Generate Semantic Label ---
    label = "objects" # default
    
    if func_name == "scene":
        label = "all objects in the scene"
        
    elif func_name == "filter":
        # args[0] is input objects, args[1] is filter criteria name
        prev_label = arg_labels[0]
        criteria = args[1].replace("_", " ")
        
        # heuristic: if filtering 'all objects', just say "apple objects"
        # if filtering something specific, say "red apple objects"
        if "all objects" in prev_label:
            label = f"{criteria} object(s)"
        else:
            # e.g., "apple objects that are red"
            label = f"{prev_label} that are {criteria}" 

    elif func_name == "relate":
        # arg_labels[0] = target set, arg_labels[1] = reference set
        target_desc = arg_labels[0]
        ref_desc = arg_labels[1]
        relation = args[2]
        label = f"{target_desc} that are {relation} the {ref_desc}"

    elif func_name == "relate_multi":
        label = f"{arg_labels[0]} that are {args[3]} between the {arg_labels[1]} and the {arg_labels[2]}"

    elif func_name == "relate_anchor":
        label = f"{arg_labels[0]} that are {args[3]} the {arg_labels[2]} when facing the {arg_labels[1]}"

    elif func_name == "union":
        label = f"objects that are either {arg_labels[0]} or {arg_labels[1]}"

    elif func_name in ["intersection", "intersect"]:
        label = f"objects that are both {arg_labels[0]} and {arg_labels[1]}"

    elif func_name == "exclude":
        label = f"{arg_labels[0]} except for the {arg_labels[1]}"
    
    # --- 2. Execution Logic (Existing + return label) ---
    if dummy_execution:
        # In dummy execution, we don't need real results, just a placeholder.
        # The placeholder should be a list to be compatible with downstream processing.
        # We still need to calculate the category.
        dummy_result = [{"id": -1, "name": "dummy_object"}]
        if func_name == "scene":
            _, category = execute_scene(args, scene_data)
            return dummy_result, category
        elif func_name == "filter":
            _, category = execute_filter(args, scene_data)
            return dummy_result, category
        elif func_name == "relate":
            _, category = execute_relate(args, arg_categories, scene_data, target_object_id)
            return dummy_result, category
        elif func_name == "relate_multi" or func_name == "relate_anchor":
            _, category = execute_relate_multi(args, arg_categories, scene_data, target_object_id)
            return dummy_result, category
        elif func_name == "union" or func_name in ["intersection", "intersect"]:
            _, category = execute_set_operation(func_name, args, arg_categories, scene_data)
            return dummy_result, category
        elif func_name == "exclude":
            _, category = execute_exclude(args, arg_categories, scene_data)
            return dummy_result, category
        else:
            raise ValueError(f"Unknown function: {func_name}")


    # if func_name == "scene":
    #     return execute_scene(args, scene_data)
    # elif func_name == "filter":
    #     return execute_filter(args, scene_data)
    # elif func_name == "relate":
    #     return execute_relate(args, arg_categories, scene_data, target_object_id)
    # elif func_name == "relate_multi" or func_name == "relate_anchor":
    #     return execute_relate_multi(args, arg_categories, scene_data, target_object_id)
    # elif func_name == "union" or func_name in ["intersection", "intersect"]:
    #     return execute_set_operation(func_name, args, arg_categories, scene_data)
    # elif func_name == "exclude":
    #     return execute_exclude(args, arg_categories, scene_data)
    # else:
    #     raise ValueError(f"Unknown function: {func_name}")

    # Real execution
    if func_name == "scene":
        res, cat = execute_scene(args, scene_data)
    elif func_name == "filter":
        res, cat = execute_filter(args, scene_data)
    elif func_name == "relate":
        res, cat = execute_relate(args, arg_categories, scene_data, target_object_id)
    elif func_name in ["relate_multi", "relate_anchor"]:
        res, cat = execute_relate_multi(args, arg_categories, scene_data, target_object_id)
    elif func_name == "union" or func_name in ["intersection", "intersect"]:
        res, cat = execute_set_operation(func_name, args, arg_categories, scene_data)
    elif func_name == "exclude":
        res, cat = execute_exclude(args, arg_categories, scene_data)
    else:
        raise ValueError(f"Unknown function: {func_name}")
        
    return res, cat, label

def execute_scene(args, scene_data):
    """Return all objects in the scene"""
    return [obj for obj in scene_data["objects"]], "some object"

def execute_filter(args, scene_data):
    """Filter objects based on property"""
    objects = args[0]  # Should be scene() output
    property_name = args[1]  # Category to filter by
    
    # Replace underscores with spaces in property name
    property_name = property_name.replace("_", " ")
    
    filtered_objects = [obj for obj in objects if obj["name"].lower() == property_name.lower()]
    return filtered_objects, property_name

def execute_relate(args, arg_categories, scene_data, target_object_id=None):
    """Find objects with specific spatial relation to reference objects"""
    target_objects = args[0]  # Objects to check
    reference_objects = args[1]  # Reference objects
    relation = args[2]  # Relation type
    
    # We assume the target object ALWAYS has the relation and others NEVER have it
    related_objects = []
    
    for target in target_objects:
        # If target_object_id is specified, only the target object has the relation
        if target_object_id is not None and str(target["id"]) == str(target_object_id):
            # Include target object if there's at least one reference object to relate to
            if reference_objects:
                related_objects.append(target)
        else:
            # Only check real spatial relations for realistic thinking trace
            for reference in reference_objects:
                # But in final results, we enforce our assumption
                if target_object_id is None and check_spatial_relation(target, reference, relation):
                    related_objects.append(target)
                    break
    
    # The category of the result is the category of the target objects
    category = arg_categories[0]
    return related_objects, category


def execute_relate_multi(args, arg_categories, scene_data, target_object_id=None):
    """
    Find objects with specific spatial relation to multiple reference objects
    e.g., 
        - relate_anchor(A, B, C, left) -> facing B, find A left to C
        - relate_multi(A, B, C, middle) -> find A in the middle of B and C
    """
    target_objects = args[0]  # Objects to check
    reference_objects = args[1]  # Reference objects
    reference_objects_2 = args[2]  # Second set of reference objects
    relations = args[3]  # Relation types
    
    # We assume the target object ALWAYS has the relation and others NEVER have it
    related_objects = []
    
    for target in target_objects:
        # If target_object_id is specified, only the target object has the relation
        if target_object_id is not None and str(target["id"]) == str(target_object_id):
            # Include target object if there's at least one reference object to relate to
            if reference_objects and reference_objects_2:
                related_objects.append(target)

    category = arg_categories[0]

    return related_objects, category

def execute_set_operation(operation, args, arg_categories, scene_data):
    """Execute set operations (union or intersection)"""
    set1 = args[0]
    set2 = args[1]
    cat1 = arg_categories[0]
    cat2 = arg_categories[1]
    
    if operation == "union":
        # Use object ID(s) to ensure uniqueness
        result = list(set1)
        existing_ids = set(obj["id"] for obj in result)
        
        for obj in set2:
            if obj["id"] not in existing_ids:
                result.append(obj)
                existing_ids.add(obj["id"])
        
        category = cat1 if cat1 == cat2 else f"({cat1} or {cat2})"
    elif operation in ["intersection", "intersect"]:
        # Use object ID(s) for intersection
        ids2 = set(obj["id"] for obj in set2)
        result = [obj for obj in set1 if obj["id"] in ids2]
        category = cat1 if cat1 == cat2 else f"({cat1} and {cat2})"
    
    return result, category

def execute_exclude(args, arg_categories, scene_data):
    """Exclude objects in set2 from set1"""
    set1 = args[0]
    set2 = args[1]
    cat1 = arg_categories[0]
    cat2 = arg_categories[1]
    
    # Use object ID(s) for exclusion
    ids2 = set(obj["id"] for obj in set2)
    result = [obj for obj in set1 if obj["id"] not in ids2]
    
    category = f"({cat1} but not {cat2})"
    return result, category

def check_spatial_relation(obj1, obj2, relation):
    """Check if obj1 has the specified spatial relation to obj2"""
    # Get positions and sizes
    pos1 = obj1["location"]
    pos2 = obj2["location"]
    size1 = obj1["size"]
    size2 = obj2["size"]
    
    # Calculate distance and centers
    distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)**0.5
    
    # Simple rules for relations (approximate values)
    if relation == "near":
        max_dimension = max(max(size1), max(size2))
        return distance < max_dimension * 3  # Within 3x size range
    elif relation == "far":
        max_dimension = max(max(size1), max(size2))
        return distance > max_dimension * 5  # Beyond 5x size range
    elif relation == "left":
        return pos1[0] < pos2[0]
    elif relation == "right":
        return pos1[0] > pos2[0]
    elif relation == "above" or relation == "over":
        return pos1[1] > pos2[1]
    elif relation == "below" or relation == "under" or relation == "beneath" or relation == "underneath":
        return pos1[1] < pos2[1]
    elif relation == "in_front" or relation == "front":
        return pos1[2] > pos2[2]
    elif relation == "behind" or relation == "back":
        return pos1[2] < pos2[2]
    elif relation == "beside" or relation == "next_to" or relation == "next":
        max_dimension = max(max(size1), max(size2))
        return distance < max_dimension * 2 and abs(pos1[1] - pos2[1]) < max_dimension
    elif relation == "on":
        return (abs(pos1[0] - pos2[0]) < (size1[0] + size2[0])/2 and 
                abs(pos1[2] - pos2[2]) < (size1[2] + size2[2])/2 and
                abs(pos1[1] - (pos2[1] + size2[1]/2 + size1[1]/2)) < 0.1)
    else:
        # Default to False for unknown relations
        return False

# --- Speedup: lightweight word counting (replace nltk.word_tokenize) ---
_WORD_RE = re.compile(r"\w+")
def _fast_word_count(text: str) -> int:
    # simple regex token count; ~10-50x faster than nltk.word_tokenize for our use
    return len(_WORD_RE.findall(text))
# --- end speedup block ---

def generate_thinking_trace(
    execution_trace, 
    scene_data, 
    nl_description, 
    target_object_id=None, 
    reference_object_ids: dict[int, str] = None, 
    use_full_detail_for_filter: bool = False, 
    only_add_positive_relations: bool = False, 
    add_plans_first: bool = False, 
    only_plans: bool = False,
    precision: Optional[int] = None
):
    """
    Generate human-readable thinking trace from execution trace using Semantic Labels.
    """
    
    # [Apeiria's Dynamic Formatter]
    # Create a format string for nested f-string usage. e.g., ".2f" or ".4f"
    if precision is None:
        precision = Templates.DEFAULT_PRECISION
    fmt = f".{precision}f"

    thinking_lines = ["[APEIRIA THINKS]"]
    thinking_lines.append(f"I need to find the object described as: \"{nl_description}\"")

    plans = ["Let's plan my next steps: "]

    # Track processed function calls to avoid duplicates
    processed_calls = set()

    for i, trace_entry in enumerate(execution_trace):
        func = trace_entry["function"]
        args = trace_entry["arguments"]
        result = trace_entry["result"]
        
        # --- NEW: Extract Semantic Labels ---
        # Fallback to "objects" if label is missing (backward compatibility)
        current_label = trace_entry.get("result_label", "objects")
        arg_labels = trace_entry.get("arg_labels", [])
        
        # Helper to safely get arg labels
        def get_arg_label(idx, default="some objects"):
            if idx < len(arg_labels) and arg_labels[idx]:
                return arg_labels[idx]
            return default

        # --- Deduplication Logic ---
        call_signature = [func]
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], dict) and 'id' in arg[0]:
                # For object lists, use a representation of their IDs
                arg_ids = sorted([obj['id'] for obj in arg if 'id' in obj])
                call_signature.append(f"list:{len(arg)}:{','.join(map(str, arg_ids))}")
            else:
                # For simple args
                call_signature.append(str(arg))
        
        call_id = "|".join(call_signature)
        
        # Skip if we've already processed this exact function call
        if call_id in processed_calls:
            continue
        
        # Mark this call as processed
        processed_calls.add(call_id)
        
        # --- Logic Generation per Function ---

        if func == "scene":
            if not only_plans:
                thinking_lines.append(f"First, I'll examine all {len(result)} objects in the scene.")
                object_id_class_info = ", ".join(f"{obj['id']} ({obj['name']})" for obj in result)
                thinking_lines.append(f"I see {len(result)} object(s) in the scene: {object_id_class_info}")

            plans.append(f"Examine all objects in the scene")
        
        elif func == "filter":
            # args[1] is the filter property (e.g., "chair")
            object_class = args[1].replace("_", " ")
            input_desc = get_arg_label(0)

            if not only_plans:
                # Use semantic label: "Looking in [all objects] to find [tables]"
                thinking_lines.append(f"Looking in {input_desc} to find {object_class}.")
                
                if len(result) == 0:
                    thinking_lines.append(f"Found none.")
                else:
                    if use_full_detail_for_filter:
                        # use full object detail for filter, including location and size
                        object_id_details = []
                        for obj in result:
                            location = obj.get("location", [0, 0, 0])
                            size = obj.get("size", [0, 0, 0])
                            # [Apeiria Modified] Use nested formatting {val:{fmt}}
                            object_id_details.append(f"{obj['id']} ({obj['name']}) at ({location[0]:{fmt}}, {location[1]:{fmt}}, {location[2]:{fmt}}), size {size[0]:{fmt}} x {size[1]:{fmt}} x {size[2]:{fmt}}")

                        object_id_info = ", ".join(object_id_details)
                    else:
                        object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)

                    thinking_lines.append(f"Found {len(result)} object(s): {object_id_info}")

            plans.append(f"Find all objects of category '{object_class}'")
                
        elif func == "relate":
            relation = args[2]
            target_objects = args[0]
            reference_objects = args[1]

            # --- Original Structure: Analyze classes and IDs ---
            target_class = target_objects[0]["name"] if target_objects else "unknown"
            ref_class = reference_objects[0]["name"] if reference_objects else "unknown"
            
            if not only_plans:
                target_id_info = f"{target_class}: " + ", ".join(str(obj["id"]) for obj in target_objects)
                ref_id_info = f"{ref_class}: " + ", ".join(str(obj["id"]) for obj in reference_objects)

                thinking_lines.append(f"Now, I'll check which {target_class}(s) are '{relation}' the {ref_class}(s).")
                thinking_lines.append(f"Analyzing {len(target_objects)} potential target object(s) ({target_id_info}) and {len(reference_objects)} reference object(s) ({ref_id_info}):")
                
                # Detailed analysis of relation matches
                assert target_object_id is not None and reference_object_ids is not None, "Target and reference object ID(s) must be specified for relation analysis"
                # for binary relations, reference_object_ids should have only one entry
                reference_object_id, reference_object_class = list(reference_object_ids.items())[0]
                for target in target_objects:
                    for reference in reference_objects:
                        is_relation = (str(target["id"]) == str(target_object_id) and str(reference["id"]) == str(reference_object_id))
                        
                        if only_add_positive_relations and not is_relation:
                            continue
                        
                        relation_text = "is" if is_relation else "is not"
                        # Original line structure
                        thinking_lines.append(f"Object {target['id']} ({target['name']}) {relation_text} {relation} to Object {reference['id']} ({reference['name']}).")

                
                object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)
                thinking_lines.append(f"After analysis, found {len(result)} object(s) with the '{relation}' relation: {object_id_info}")
            
            # For plan, use the semantic label (better context)
            target_desc_label = get_arg_label(0) if target_class == "unknown" else target_class + "(s)"
            ref_desc_label = get_arg_label(1) if ref_class == "unknown" else ref_class + "(s)"
            plans.append(f"Check which {target_desc_label} are '{relation}' the {ref_desc_label}")

        elif func == "relate_multi" or func == "relate_anchor":
            relation = args[3]
            target_objects = args[0]
            reference_objects = args[1]
            reference_objects_2 = args[2]

            target_class = target_objects[0]["name"] if target_objects else "unknown"
            ref_class = reference_objects[0]["name"] if reference_objects else "unknown"
            ref_class_2 = reference_objects_2[0]["name"] if reference_objects_2 else "unknown"

            if not only_plans:
                # --- Original Structure ---
                target_id_info = f"{target_class}: " + ", ".join(str(obj["id"]) for obj in target_objects)
                ref_id_info = f"{ref_class}: " + ", ".join(str(obj["id"]) for obj in reference_objects)
                ref_id_info_2 = f"{ref_class_2}: " + ", ".join(str(obj["id"]) for obj in reference_objects_2)

                thinking_lines.append(f"Analyzing {len(target_objects)} potential target object(s) ({target_id_info}), {len(reference_objects)} reference object(s) ({ref_id_info}), and {len(reference_objects_2)} second reference object(s) ({ref_id_info_2}).")

                # Detailed analysis of relation matches
                assert target_object_id is not None and reference_object_ids is not None, "Target and reference object ID(s) must be specified for relation analysis"

                # reference_object_ids is an object_id -> object_class mapping
                # we need to according to the ref_class and ref_class_2, find the corresponding object_id of correct reference object
                reference_object_class_to_id = {v: k for k, v in reference_object_ids.items()}
                
                reference_object_id = reference_object_class_to_id.get(ref_class)
                reference_object_id_2 = reference_object_class_to_id.get(ref_class_2)

                for target in target_objects:
                    for reference in reference_objects:
                        for reference_2 in reference_objects_2:
                            is_relation = (str(target["id"]) == str(target_object_id) and 
                                           str(reference["id"]) == str(reference_object_id) and
                                           str(reference_2["id"]) == str(reference_object_id_2))
                            
                            relation_text = "is" if is_relation else "is not"

                            if func == "relate_multi":
                                thinking_lines.append(f"Object {target['id']} ({target['name']}) {relation_text} {relation} to Object {reference['id']} ({reference['name']}) and {reference_2['id']} ({reference_2['name']}).")
                            elif func == "relate_anchor":
                                thinking_lines.append(f"Facing Object {reference['id']} ({reference['name']}), Object {target['id']} ({target['name']}) {relation_text} {relation} to Object {reference_2['id']} ({reference_2['name']}).")

                object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)
                thinking_lines.append(f"After analysis, found {len(result)} object(s) with the '{relation}' relation: {object_id_info}")

            # For plan, use semantic labels
            target_desc_label = get_arg_label(0) if target_class == "unknown" else target_class + "(s)"
            ref_desc_label = get_arg_label(1) if ref_class == "unknown" else ref_class + "(s)"
            ref_desc_label_2 = get_arg_label(2) if ref_class_2 == "unknown" else ref_class_2 + "(s)"
            
            if func == "relate_multi":
                thinking_lines.append(f"Now, I'll check which {target_desc_label} are '{relation}' to the {ref_desc_label} and {ref_desc_label_2}.")
                plans.append(f"Check which {target_desc_label} are '{relation}' to the {ref_desc_label} and {ref_desc_label_2}")
            elif func == "relate_anchor":
                thinking_lines.append(f"Now, I'll check which {target_desc_label} are '{relation}' to {ref_desc_label_2} when facing the {ref_desc_label}.")
                plans.append(f"Check which {target_desc_label} are '{relation}' to {ref_desc_label_2} when facing the {ref_desc_label}")

        elif func == "union":
            desc_a = get_arg_label(0)
            desc_b = get_arg_label(1)
            
            if not only_plans:
                thinking_lines.append(f"Combining {desc_a} and {desc_b}.")
                object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)
                thinking_lines.append(f"The union contains {len(result)} objects: {object_id_info}")
            
            plans.append(f"Combine {desc_a} and {desc_b}")
        
        elif func in ["intersection", "intersect"]:
            desc_a = get_arg_label(0)
            desc_b = get_arg_label(1)
            
            if not only_plans:
                thinking_lines.append(f"Finding objects that are both {desc_a} AND {desc_b}.")
                object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)
                thinking_lines.append(f"The intersection contains {len(result)} objects: {object_id_info}")
            
            plans.append(f"Find overlap of {desc_a} and {desc_b}")
        
        elif func == "exclude":
            desc_a = get_arg_label(0)
            desc_b = get_arg_label(1)
            
            if not only_plans:
                thinking_lines.append(f"From {desc_a}, excluding {desc_b}.")
                object_id_info = "ID(s): " + ", ".join(str(obj["id"]) for obj in result)
                thinking_lines.append(f"After exclusion, {len(result)} objects remain: {object_id_info}")
            
            plans.append(f"Exclude {desc_b} from {desc_a}")

        else:
            raise ValueError(f"Unknown function: {func}")
    
    # --- Conclusion ---
    if execution_trace and not only_plans:
        final_result = execution_trace[-1]["result"]
        if len(final_result) == 0:
            thinking_lines.append("In conclusion: I didn't find any objects matching the query.")
        else:
            thinking_lines.append(f"In conclusion: I found {len(final_result)} object(s) matching the query:")
            for obj in final_result:
                # [Apeiria Modified] Use nested formatting {val:{fmt}}
                thinking_lines.append(f"Object {obj['id']}: {obj['name']} at ({obj['location'][0]:{fmt}}, {obj['location'][1]:{fmt}}, {obj['location'][2]:{fmt}}), size {obj['size'][0]:{fmt}} x {obj['size'][1]:{fmt}} x {obj['size'][2]:{fmt}}")
    
    thinking_lines.append("[APEIRIA SPEAKS]")

    # Organize parts for return
    plans_text = plans[0] + "; ".join(plans[1:])
    header = thinking_lines[:2]
    executions = thinking_lines[2:]
    tailer = [thinking_lines[-1]]
    
    if add_plans_first:
        thinking_lines = header + [plans_text] + executions

    parts_dict = {
        "all": copy.deepcopy(thinking_lines),
        "plan": copy.deepcopy(plans_text),
        "execution": "\n".join(executions),
        "header": "\n".join(header),
    }

    if only_plans:
        # NOTE: no need to add the execution trace (and the [APEIRIA speaks] trailer]), let the model generate it - and conduct RL based on the result.
        thinking_lines = header + [plans_text] + tailer

    return "\n".join(thinking_lines), parts_dict

def process_annotation_chunk(chunk_info, instruction_templates, 
                            object_detail_templates, response_templates, 
                            add_thinking_trace, name, split, tokenizer, 
                            dummy_execution=False, is_m3dref=False, add_bracket_in_object_detail=False,
                            **generate_kwargs):
    """
    Worker function to process a chunk of annotations in parallel.
    chunk_info shall be a tuple (annotation, start_index) where:
    - start_index is the index of the annotation in the original dataset
    """
    global _worker_scene_data

    chunk_samples = []
    
    chunk, start_idx = chunk_info

    if add_bracket_in_object_detail:
        # modify object detail templates to add brackets
        # so that we could locate them easily in the response
        # NOTE: add space, to ensure "[" and "]" are tokenized separately
        # object_detail_templates_for_response = [f"[{tpl}]" for tpl in object_detail_templates]
        object_detail_templates_for_response = []
        for tpl in object_detail_templates:
            tpl_prefix, tpl_suffix = tpl.split(": ", 1)
            object_detail_templates_for_response.append(f"{tpl_prefix}: [{tpl_suffix}]")

    else:
        object_detail_templates_for_response = object_detail_templates
        
    
    for idx, anno in enumerate(chunk):
        sample_idx = start_idx + idx
        try:
            scene_id = anno["scene_id"]
            
            # if scene_id not in scene_data:
            if scene_id not in _worker_scene_data:
                continue

            # TODO: handle multi3dref: can have no target object or many target objects, specified by object_id list
            if is_m3dref:
                object_ids = anno["object_ids"]
                object_id = ",".join([str(oid) for oid in object_ids]) if len(object_ids) > 0 else "none"
            else:
                object_id = int(anno.get("object_id"))
                object_ids = [object_id]
                
            description = anno["description"]
            # program = anno["program"]
            program = anno.get("program", "")
            ref_object_ids = anno.get("anchor_ids", [])
            ref_object_ids = [int(ref_id) for ref_id in ref_object_ids]
            ref_object_classes = anno.get("anchors_types", [])
            ref_object_id_to_class = dict(zip(ref_object_ids, ref_object_classes))
            
            # Get objects for this scene
            # objects = scene_data[scene_id]["objects"]
            objects = _worker_scene_data[scene_id]["objects"]
            # logger.info(objects)
            
            # find the target object index
            # target_object = None
            # for obj in objects:
            #     if str(obj["id"]) == str(object_id):
            #         target_object = obj
            #         break
            target_object = []
            for obj in objects:
                if str(obj["id"]) in [str(oid) for oid in object_ids]:
                    target_object.append(obj)

            if len(target_object) == 0 and not is_m3dref:
                logger.warning(f"Target object not found in scene: {scene_id}, object ID: {object_id}. Maybe it was filtered out?")
                continue

            target_object_name = target_object[0]["name"] if len(target_object) > 0 else "none"

            # Create scan2cap_id and scanrefer_id
            scan2cap_id = f"{scene_id}|{object_id}|{target_object_name}" if object_id is not None else f"{scene_id}|unknown|unknown"
            scanrefer_id = f"{scene_id}|{sample_idx}"
            hash_id = f"real_{scene_id}_{sample_idx}"
            
            # Create input prompt
            prompt = random.choice(instruction_templates).format(description=description)
            
            if add_thinking_trace:
                # Parse program and generate execution trace
                try:
                    # Parse program to AST
                    ast = parse_program(program, {"objects": objects})
                    
                    # Execute program to get results and execution trace
                    result, _, _, execution_trace = execute_program(ast, {"objects": objects}, object_id, dummy_execution=dummy_execution)
                    
                    # Generate thinking trace text
                    thinking_trace, parts = generate_thinking_trace(execution_trace, {"objects": objects}, 
                                                            description, object_id, ref_object_id_to_class, 
                                                            **generate_kwargs,
                                                            )

                    # Generate expected response
                    # if result: # generate response even if no result, to inform no object found

                    if not dummy_execution:
                        assert len(result) > 0, f"Empty result for program: {program}"

                        assert str(result[0]["id"]) == str(object_id), (
                            f"First object in result must match target object ID"
                            f", but got {result[0]['id']}(name: {result[0]['name']}) vs GT:{object_id}"
                        )
                    else:
                        # we use GT target object as the result
                        # result = [target_object]
                        result = target_object

                    object_details = []
                    for obj in result:
                        location = obj.get("location", [0, 0, 0])
                        size = obj.get("size", [0, 0, 0])
                        object_details.append(
                            random.choice(object_detail_templates_for_response).format(
                                id=obj["id"], 
                                x=location[0],
                                y=location[1],
                                z=location[2],
                                width=size[0],
                                height=size[1],
                                depth=size[2]
                            )
                        )
                    
                    response_body = random.choice(response_templates).format(
                        count=len(result),
                        object_details="\n".join(object_details) if object_details else Templates.NO_OBJECT_FOUND_RESPONSE
                    )

                    if add_thinking_trace:
                        expected_response = thinking_trace + "\n" + response_body
                    else:
                        expected_response = response_body

                    # else:
                    #     logger.warning(f"No objects found for program: {program}")
                    #     response_body = "Apeiria didn't find any objects matching the description."
                    #     expected_response = response_body if not add_thinking_trace else thinking_trace + "\n" + response_body
                except KeyError as e:
                    logger.warning(f"Error processing program: {program}, error: {str(e)}")
                    logger.warning(f"Scene ID: {scene_id}, object ID: {object_id}, description: {description}") 
                    # NOTE: in Sr3D, two annotations seems to be out of vanish, so we skip them
                    # logger.warning(f"Full scene data: {scene_data[scene_id]}")
                    print_exc()
                    # raise e
                    continue

                except lark.exceptions.LarkError as e:
                    # wrong program, skip FIXME: make no such error
                    logger.warning(f"Unexpected characters in program: {program}")
                    logger.warning(f"Scene ID: {scene_id}, object ID: {object_id}, description: {description}")
                    print_exc()
                    continue

                except Exception as e:
                    # print current program and description
                    # still, wrong program, skip
                    logger.warning(f"Error processing program: {program}, error: {str(e)}")
                    logger.warning(f"Scene ID: {scene_id}, object ID: {object_id}, description: {description}")
                    print_exc()
                    # raise e
                    continue

            else:
                # Use target object details directly
                # result = [target_object]
                result = target_object
                object_details = []
                for obj in result:
                    location = obj.get("location", [0, 0, 0])
                    size = obj.get("size", [0, 0, 0])
                    object_details.append(
                        random.choice(object_detail_templates_for_response).format(
                            id=obj["id"],
                            x=location[0],
                            y=location[1],
                            z=location[2],
                            width=size[0],
                            height=size[1],
                            depth=size[2]
                        )
                    )
                
                response_body = random.choice(response_templates).format(
                    count=len(result),
                    object_details="\n".join(object_details) if object_details else Templates.NO_OBJECT_FOUND_RESPONSE
                )
                expected_response = response_body
                parts = None

            # stat prompt and response total words
            # prompt_words = len(nltk.word_tokenize(prompt))
            # response_words = len(nltk.word_tokenize(expected_response))
            prompt_words = _fast_word_count(prompt)
            response_words = _fast_word_count(expected_response)

            # Create the instruction for RL - includes the prompt and the plan
            # NOTE: we need to follow the chat template
            if add_thinking_trace:
                plan_with_start = parts["header"] + "\n" + parts["plan"]
                prompt_with_plan = apply_qwen_template_with_partial_response(prompt, tokenizer, plan_with_start)[0]
            else:
                prompt_with_plan = None
            
            # Store sample
            chunk_samples.append({
                # "prompt": prompt,
                # "answer": expected_response,
                "description": prompt,
                "prompt_with_plan": prompt_with_plan,
                "raw_description": description,
                "program": program,
                "scene_id": scene_id,
                "object_id": object_id,
                # "object_ids": [object_id],
                "object_ids": object_ids,
                "result_objects": result,
                "ann_id": anno["ann_id"],
                "objects": result,  # For reward calculation compatibility
                "question_id": f"{scene_id}_{sample_idx}",
                "raw_question_id": f"{scene_id}_{sample_idx}",
                "scan2cap_id": scan2cap_id,
                "scanrefer_id": scanrefer_id,
                "hash_id": hash_id,
                "data_type": name,
                "split": split,
                "expected_response": expected_response,
                "prompt_words": prompt_words,
                "response_words": response_words,
                "all_words": prompt_words + response_words,
                "thinking_trace_parts": parts,
            })
        except Exception as e:
            logger.warning(f"Error processing annotation {sample_idx}: {str(e)}")
            print_exc()
            raise e
            continue
            
    return chunk_samples

def parse_response(response: str):
    """
    Parse the model's response to extract object ID(s) and locations.
    
    Args:
        response: String response from the model
        
    Returns:
        List of dicts with keys: id, x, y, z, width, height, depth
    """
    parsed_objects = []

    # Remove thinking trace if present
    if re.search(r"\[APEIRIA THINKS\]", response, flags=re.IGNORECASE) is not None:
        if re.search(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE) is None:
            logger.warning("Thinking trace detected without response, skipping...")
            logger.warning(f"Response: {response}")
            return parsed_objects
        else:
            # Remove all contents before [APEIRIA SPEAKS]
            response = re.split(r"\[APEIRIA SPEAKS\]", response, flags=re.IGNORECASE)[-1].strip()

    # remove "brackets" if any, they may interfere with parsing
    response = response.replace("[", "").replace("]", "")
    
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

    # deduplicate parsed objects by ID, since one line may match multiple patterns
    unique_objects = {}
    for obj in parsed_objects:
        unique_objects[obj["id"]] = obj
    parsed_objects = list(unique_objects.values())

    # if not parsed_objects:
    #     logger.warning(f"No objects found in response: {response}")
    
    return parsed_objects

# tools to normalize scan2cap captions on eval
def preprocess_sos_eos_for_scan2cap(text: str) -> str:
    if not text.startswith("sos"):
        text = "sos " + text.strip()

    if not text.endswith("eos"):
        text = text.strip() + " eos"

    return text

def postprocess_punctuation_for_caption_metrics(text: str) -> str:
    # add back space before punctuation
    punctuation_chars = [".", ",", "!", "?", ":", ";", "-", "'", "\"", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\"]
    for p in punctuation_chars:
        text = text.replace(p, f" {p}")

    # remove double spaces
    text = text.replace("  ", " ")
    return text

class Templates:
    """Class to store and manage instruction, response, and object detail templates."""
    NO_OBJECT_FOUND_RESPONSE = "No objects found."
    DEFAULT_PRECISION = 2 # Can be configured outside

    @classmethod
    def get_templates(cls, **kwargs):
        """a factory to return a new instance of Templates"""
        templates = cls()
        templates.fix_template = kwargs.get("fix_template", False)
        templates.add_thinking_trace = kwargs.get("add_thinking_trace", False)
        templates.precision = kwargs.get("precision", cls.DEFAULT_PRECISION)
        templates._initialize_templates(**kwargs)
        return templates

    def _init_args_if_missing(self, **kwargs):
        """Initialize any missing arguments with default values."""
        # Check kwargs, if missing, use existing attribute or default value
        self.fix_template = kwargs.get("fix_template", getattr(self, "fix_template", False))
        self.add_thinking_trace = kwargs.get("add_thinking_trace", getattr(self, "add_thinking_trace", False))
        self.add_thinking_trace_prompt = kwargs.get("add_thinking_trace_prompt", getattr(self, "add_thinking_trace_prompt", False))
        self.precision = kwargs.get("precision", getattr(self, "precision", self.DEFAULT_PRECISION))

    def _initialize_templates(self, **kwargs):
        """Initialize templates for instructions, responses, etc."""
        self._init_args_if_missing(**kwargs)

        # 动态构建格式化后缀，例如: ":.4f"
        p_fmt = f":.{self.precision}f"

        # Instruction templates
        self.instruction_templates = kwargs.get("instruction_templates", None) or [
            "Identify the object described as: \"{description}\". Respond with the object's ID, position, and size.\n",
            "Find the object described as: \"{description}\". Provide the object's ID, location, and dimensions.\n",
            "{description}. Where is it? Give me its ID(s), positions, and dimensions.\n",
            "Locate the object: \"{description}\". Specify its ID, location, and size.\n",
            "Tell me where the object described as: \"{description}\" is. Provide its ID, position, and dimensions.\n"
        ]

        # add multi3Dref specific instruction templates - they have 0 or one or many target objects
        # NOTE: for old checkpoints, this does not exist, and these models does not support such prompts (maybe work, but not optimized for that)
        if getattr(self, 'is_m3dref', False):
            self.instruction_templates = kwargs.get("m3dref_instruction_templates", None) or [
                "Identify all object(s) described as, if any: \"{description}\". Respond with each object's ID, position, and size.\n",
                "Find all (if any) object(s) described as: \"{description}\". Provide each object's ID, location, and dimensions.\n",
                "{description}. Where is/are it/they (if any)? Give me their ID(s), positions, and dimensions.\n",
                "Locate all object(s) if exists: \"{description}\". Specify their IDs, locations, and sizes.\n",
                "Tell me where the object(s) (if exsits) described as: \"{description}\" is/are. Provide their IDs, positions, and dimensions.\n"
            ]

            # TO-ADD in new PT
            # if not self.sft: # for now, only add to RL finetuning
            self.instruction_templates = [
                tpl.replace("\n", " ") + 
                "There might be zero, one or many. If no such object exists, respond with: 'Apeiria didn't find any objects matching the description.'\n" 
                for tpl in self.instruction_templates
            ]

        # If require thinking, add a prefix to the instruction
        self.require_thinking_templates = kwargs.get("require_thinking_templates", None) or [
            "Think about the scene first. ",
            "Analyze the scene first before answering. ",
            "Consider the scene layout first. ",
            "Take a moment to understand the scene before proceeding. ",
        ]

        if self.fix_template:
            self.require_thinking_templates = [self.require_thinking_templates[0]]

        self.object_templates = "These are all objects in the scene: |object_set| \n"
        self.instruction_templates = [self.object_templates + inst for inst in self.instruction_templates]

        if self.add_thinking_trace or self.add_thinking_trace_prompt:
            # Add as prefix, so the model will not confuse between instruction with and without thinking trace
            self.instruction_templates = [random.choice(self.require_thinking_templates) + inst for inst in self.instruction_templates]
        
        # Object detail templates
        # self.object_detail_templates = kwargs.get("object_detail_templates", None) or [
        #     "Object {id}: At ({x:.2f}, {y:.2f}, {z:.2f}), size: {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "ID {id}: Position ({x:.2f}, {y:.2f}, {z:.2f}), size {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "{id}: Coordinates ({x:.2f}, {y:.2f}, {z:.2f}), dimensions {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "Object {id}: ({x:.2f}, {y:.2f}, {z:.2f}), {width:.2f} x {height:.2f} x {depth:.2f}"
        # ]

        # self.object_detail_with_class_templates = kwargs.get("object_detail_with_class_templates", None) or [
        #     "Object {id}: {object_name} at ({x:.2f}, {y:.2f}, {z:.2f}), size: {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "ID {id}: {object_name}. Position ({x:.2f}, {y:.2f}, {z:.2f}), size {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "{id}: {object_name}. Coordinates ({x:.2f}, {y:.2f}, {z:.2f}), dimensions {width:.2f} x {height:.2f} x {depth:.2f}",
        #     "Object {id}({object_name}): ({x:.2f}, {y:.2f}, {z:.2f}), {width:.2f} x {height:.2f} x {depth:.2f}"
        # ]
        # 使用 f-string 注入 p_fmt，并使用双花括号 {{}} 转义原本的占位符
        self.object_detail_templates = kwargs.get("object_detail_templates", None) or [
            f"Object {{id}}: At ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), size: {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"ID {{id}}: Position ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), size {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"{{id}}: Coordinates ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), dimensions {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"Object {{id}}: ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}"
        ]

        self.object_detail_with_class_templates = kwargs.get("object_detail_with_class_templates", None) or [
            f"Object {{id}}: {{object_name}} at ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), size: {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"ID {{id}}: {{object_name}}. Position ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), size {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"{{id}}: {{object_name}}. Coordinates ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), dimensions {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}",
            f"Object {{id}}({{object_name}}): ({{x{p_fmt}}}, {{y{p_fmt}}}, {{z{p_fmt}}}), {{width{p_fmt}}} x {{height{p_fmt}}} x {{depth{p_fmt}}}"
        ]

        # Response templates
        self.response_templates = kwargs.get("response_templates", None) or [
            "Apeiria found {count} object(s) matching the description:\n{object_details}",
            "Roger. I've located {count} object(s) as described:\n{object_details}",
            "Roger. The object(s) you described are:\n{object_details}",
            "Apeiria has identified {count} object(s):\n{object_details}",
        ]

        # Thinking trace template
        self.thinking_trace_template = kwargs.get("thinking_trace_template", None) or [
            "[APEIRIA THINKS]\n"
            "Apeiria will now analyze the scene and identify the requested object.\n"
            "First, let me list all {object_count} objects and their details:\n"
            "{object_details_with_class}\n"
            "Now, Apeiria needs to find objects matching this description: \"{description}\"\n"
            "{thinking_trace}\n"
            "Based on the analysis, Apeiria has identified these objects: {object_ids_with_class}\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]

        self.object_id_with_class_templates = "{id} ({object_name})"

        if self.fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
            self.object_detail_templates = [self.object_detail_templates[0]]
            self.thinking_trace_template = [self.thinking_trace_template[0]]
            self.object_detail_with_class_templates = [self.object_detail_with_class_templates[0]]


class ScanNetMixin:
    """
    为 ScanNet 数据集提供实用功能的类。
    使用类常量定义全局归一化参数。
    """
    # --- 归一化相关的类常量 ---
    # ScanNet 全局坐标范围 (根据注释)
    GLOBAL_MIN_XYZ = np.array([-6.17, -8.96, -1.47])
    GLOBAL_MAX_XYZ = np.array([6.68, 9.15, 5.84])
    # ScanNet 全局物体尺寸范围 (根据注释)
    GLOBAL_MIN_SIZE = np.array([0.00, 0.00, 0.00])
    GLOBAL_MAX_SIZE = np.array([11.13, 17.16, 5.36])
    # 归一化后的目标空间大小
    ROOM_SIZE = 10.0
    # --------------------------

    # --- 物体过滤相关的类常量 ---
    BAD_OBJECTS = ["wall", "floor", "ceiling", "object"]
    # --------------------------

    def _get_scene_list(self):
        # if self.split is like "a,b,c", split into list and use all those scenes
        if ',' in self.split:
            splits = self.split.strip().split(',')
        else:
            splits = [self.split]

        splits = [s.strip() for s in splits]

        # scene_list_file = f"{DATA_PATH}/meta_data/scannetv2_{self.split}.txt"
        scene_list_files = [f"{self.data_path}/meta_data/scannetv2_{split}.txt" for split in splits]
        scene_list = []
        for scene_list_file in scene_list_files:
            if not os.path.exists(scene_list_file):
                raise FileNotFoundError(f"Scene list file not found: {scene_list_file}")
            
            with open(scene_list_file, 'r') as f:
                scene_list_split = f.read().splitlines()

            # 过滤掉空行
            scene_list_split = [scene.strip() for scene in scene_list_split if scene.strip()]
            scene_list.extend(scene_list_split)

        self.scene_list = sorted(set(scene_list))

        logger.info(f"Resolved {len(self.scene_list)} scenes for split(s) '{self.split}' of dataset {self.name} ({self.__class__.__name__})")

        return self.scene_list

    def _load_scene_data(self):
        """加载注释中所有场景的场景数据，并使用类常量进行归一化。"""
        scene_data = {}
        # 确保 self.annotations, self.data_path, self.pre_filter_objects, self.max_objects 已经设置好
        # if not hasattr(self, 'annotations') or not self.annotations:
        #      logger.error("Annotations are not loaded before calling _load_scene_data.")
        #      raise ValueError("Annotations are not loaded before calling _load_scene_data.")
        #      return scene_data
        if not hasattr(self, 'data_path'):
             logger.error("data_path attribute is not set.")
             raise ValueError("data_path must be set before calling _load_scene_data.")
             return scene_data
        if not hasattr(self, 'pre_filter_objects'):
             logger.warning("pre_filter_objects attribute not set, defaulting to False.")
             self.pre_filter_objects = False
        if not hasattr(self, 'max_objects'):
             logger.warning("max_objects attribute not set, defaulting to 50.")
             self.max_objects = 50


        # scene_ids = sorted(set(anno["scene_id"] for anno in self.annotations))
        scene_ids = self._get_scene_list()
        
        for scene_id in scene_ids:
            # scene_file = f"{self.data_path}/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed/{scene_id}.json"
            scene_file = f"{self.data_path}/apeiria_scannet_w_caption_gpt4o_and_corners_and_nyu_names_fixed_prec4/{scene_id}.json" # use precision 4 object boxes
            try:
                with open(scene_file, 'r') as f:
                    scene_info = json.load(f)

                # 存储原始对象和用于该场景的归一化参数（来自类常量）
                scene_info["original_objects"] = copy.deepcopy(scene_info["objects"])
                scene_info["global_min_xyz"] = ScanNetMixin.GLOBAL_MIN_XYZ
                scene_info["global_max_xyz"] = ScanNetMixin.GLOBAL_MAX_XYZ
                scene_info["global_min_size"] = ScanNetMixin.GLOBAL_MIN_SIZE
                scene_info["global_max_size"] = ScanNetMixin.GLOBAL_MAX_SIZE
                scene_info["normalized_room_size"] = ScanNetMixin.ROOM_SIZE

                current_objects = scene_info["objects"]

                # 预过滤对象 (如果需要)
                if self.pre_filter_objects:
                    current_objects = self._pre_filter_objects(current_objects)

                # 限制最大对象数量
                current_objects = current_objects[:self.max_objects]

                # 使用类常量进行归一化
                scene_info["objects"] = self._normalize_objects_in_scene(
                    current_objects,
                    ScanNetMixin.GLOBAL_MIN_XYZ, ScanNetMixin.GLOBAL_MAX_XYZ,
                    ScanNetMixin.GLOBAL_MIN_SIZE, ScanNetMixin.GLOBAL_MAX_SIZE,
                    ScanNetMixin.ROOM_SIZE
                )

                scene_data[scene_id] = scene_info
            except FileNotFoundError:
                logger.warning(f"场景文件未找到: {scene_file}")
            except Exception as e:
                 logger.error(f"加载或处理场景 {scene_id} 时出错: {e}")

        logger.info(f"已加载 {len(scene_data)} 个场景数据文件")
        return scene_data

    def _pre_filter_objects(self, objects):
        """预过滤掉不想要的物体类别 (使用类常量 BAD_OBJECTS)。"""
        return [obj for obj in objects if obj.get("name", "").lower() not in ScanNetMixin.BAD_OBJECTS]

    def _normalize_box(self, location, size, min_xyz, max_xyz, min_size, max_size, room_size):
        """
        将单个物体的边界框（位置和尺寸）归一化到 [0, room_size] 范围。
        (此函数内部逻辑不变，依赖传入的参数)
        """
        location_np = np.array(location)
        size_np = np.array(size)

        loc_range = np.maximum(max_xyz - min_xyz, 1e-6) # 防止除零
        size_range = np.maximum(max_size - min_size, 1e-6) # 防止除零

        normalized_location = (location_np - min_xyz) / loc_range * room_size
        normalized_size = (size_np - min_size) / size_range * room_size

        # Owner, 可以选择是否裁剪结果
        # normalized_location = np.clip(normalized_location, 0, room_size)
        # normalized_size = np.clip(normalized_size, 0, room_size)

        return normalized_location, normalized_size

    def _revert_normalized_box(self, normalized_location, normalized_size, min_xyz, max_xyz, min_size, max_size, room_size):
        """
        将归一化后的边界框（位置和尺寸）还原到原始坐标系。
        (此函数内部逻辑不变，依赖传入的参数)
        """
        norm_location_np = np.array(normalized_location)
        norm_size_np = np.array(normalized_size)

        loc_range = np.maximum(max_xyz - min_xyz, 1e-6) # 防止除零
        size_range = np.maximum(max_size - min_size, 1e-6) # 防止除零

        # 确保 room_size 不为零
        if room_size == 0:
             logger.error("Room size cannot be zero for reverting normalization.")
             # 返回一个合理的值或者抛出错误
             return np.zeros_like(norm_location_np), np.zeros_like(norm_size_np)


        original_location = (norm_location_np / room_size) * loc_range + min_xyz
        original_size = (norm_size_np / room_size) * size_range + min_size

        return original_location, original_size

    def _normalize_objects_in_scene(self, objects, min_xyz, max_xyz, min_size, max_size, room_size):
        """
        归一化场景中所有物体的边界框。
        (此函数内部逻辑不变，依赖传入的参数)
        """
        normalized_objects = []
        for obj in objects:
            if "location" in obj and "size" in obj:
                try:
                    normalized_location, normalized_size = self._normalize_box(
                        obj["location"], obj["size"],
                        min_xyz, max_xyz,
                        min_size, max_size,
                        room_size
                    )
                    # 创建一个新的 obj 字典或深拷贝以避免修改原始列表中的字典
                    new_obj = copy.deepcopy(obj)
                    new_obj["location"] = normalized_location.tolist()
                    new_obj["size"] = normalized_size.tolist()
                    normalized_objects.append(new_obj)
                except Exception as e:
                    obj_id = obj.get('id', 'N/A') # 尝试获取对象ID以便调试
                    logger.warning(f"归一化对象 (ID: {obj_id}) 时出错: {e}", exc_info=True) # 添加 exc_info 获取更详细错误
            else:
                 obj_id = obj.get('id', 'N/A')
                 logger.warning(f"对象 (ID: {obj_id}) 缺少 'location' 或 'size' 键，跳过归一化。")
                 # Owner, 如果希望保留未归一化的对象，可以取消下面这行注释
                 # normalized_objects.append(copy.deepcopy(obj))
        return normalized_objects


    def get_original_box_from_normalized(self, normalized_location, normalized_size, scene_info):
         """
         根据场景信息，将归一化的边界框还原。
         (此函数依赖 scene_info 中的参数，而 scene_info 现在由类常量填充)
         """
         try:
              # 从 scene_info 中读取保存的归一化参数
              min_xyz = np.array(scene_info["global_min_xyz"])
              max_xyz = np.array(scene_info["global_max_xyz"])
              min_size = np.array(scene_info["global_min_size"])
              max_size = np.array(scene_info["global_max_size"])
              room_size = scene_info["normalized_room_size"]

              return self._revert_normalized_box(
                   normalized_location, normalized_size,
                   min_xyz, max_xyz, min_size, max_size,
                   room_size
              )
         except KeyError as e:
              logger.error(f"场景信息中缺少必要的键用于反向归一化: {e}. Scene info keys: {scene_info.keys()}")
              return None, None
         except Exception as e:
              logger.error(f"反向归一化时发生错误: {e}", exc_info=True)
              return None, None

class DefaultViewSelectionMixin:
    """
    handles default view selection for real 3D datasets
    it is scene_id -> list of selected view indices
    
    For classes wants to use default view selection, inherit this before Real3DDataset, 
    since Real3DDataset implements its own _get_image_indices
    """

    # TODO: use real path
    def _get_view_annotation_file(self, n_views_in_m_views: int) -> str:
        # return f"{self.data_path}/data/resampledscannetv2_{self.split}_default_views.json"
        n, m = n_views_in_m_views.split('_')
        return f"{self.data_path}/resampled_views_single_scene/fps_c{n}_n{m}.json"

    def _get_image_indices(self, data):
        """Retrieve image features for a given sample."""
        scene_id = data["scene_id"]
        view_selection_id = f"{scene_id}"
        image_ids = self.view_annotation[view_selection_id]["sampled_views"]
        return image_ids

    def _get_image_features(self, scene_id: str, image_ids: List[int]) -> torch.Tensor:
        """Retrieve image features for a given scene and image IDs."""
        image_features = self.image_feature_manager.get_specific_image_features(
            scene_id=scene_id,
            image_ids=image_ids,
        ) # => [N_views, N, N, D] or [N_views, D]

        if image_features.dim() == 4:
            # [N_views, N, N, D] -> [N_views, N*N, D]
            N_views, N1, N2, D = image_features.shape
            image_features = image_features.view(N_views, N1*N2, D)
        elif image_features.dim() == 2:
            # [N_views, D], expand to [N_views, 1, D]
            N_views, D = image_features.shape
            image_features = image_features.view(N_views, 1, D)

        # pad zero if less than max views
        if self.num_views > image_features.shape[0]:
            pad_size = self.num_views - image_features.shape[0]
            image_features = torch.nn.functional.pad(image_features, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0.0)

        return image_features


def trim_proposal_features(proposal_features, trim_keys: Iterable[str]):
    # Trim masked objects, and input predicted bboxes correspondingly
    logger.info("Trimming proposal features...")
    for scene_id in proposal_features.keys():
        # print_once("Trimming frozen features...")
        object_mask = proposal_features[scene_id]["mask"]

        # don't trim objects that have no objects at all - it will cause error because frozen_features[scene_id][0] is empty
        if object_mask.sum() == 0:
            logger.warning(f"Empty mask for {scene_id}, skipping trimming. Note: the feature shall be all-zero.")
            # object_mask = torch.ones(len(object_feature), dtype=bool)
            continue

        for trim_key in trim_keys:
            if trim_key == "mask":
                continue
            if isinstance(proposal_features[scene_id][trim_key], list):
                # trim the list according to the mask
                proposal_features[scene_id][trim_key] = [proposal_features[scene_id][trim_key][i] for i in range(len(object_mask)) if object_mask[i].bool()]
            elif isinstance(proposal_features[scene_id][trim_key], torch.Tensor):
                proposal_features[scene_id][trim_key] = proposal_features[scene_id][trim_key][object_mask.bool()]
            else:
                raise TypeError(f"Unsupported type for proposal feature: {type(proposal_features[scene_id][trim_key])}")
        proposal_features[scene_id]["mask"] = torch.ones(object_mask.sum().long().item(), dtype=bool)

    return proposal_features

class Real3DDataset(Dataset, Templates, ScanNetMixin, DefaultViewSelectionMixin):
    """
    Adapter class that uses real 3D scene data (SR3D, NR3D) but follows the 
    interface and structure of the Synthetic3DDataset class.
    """
    
    def __init__(
        self,
        name: str = "sr3d",
        data_path: str = DATA_PATH,
        split: str = "train",
        ratio: float = 1.0,
        shuffle_objects: bool = False,
        start_from_last: bool = False,
        frozen_object_type: str = "synthetic",
        pc_tokenizer_type: str = "frozen",
        object_label_type: Any = ScanNetRawObject,
        max_objects: int = 100,
        max_object_id: int = 150,
        seed: int = 42,
        add_thinking_trace: bool = True,
        add_thinking_trace_prompt: bool = False, # this is need when add_thinking_trace is False, but we want the prompt to indicate thinking trace is needed
        add_full_thinking_trace_for_filter_in_relational: bool = False,
        only_add_positive_relations: bool = False,
        add_plans_first: bool = False,
        fix_template: bool = False,
        parallel: bool = True,
        load_from_cache: bool = False,
        pre_filter_objects: bool = True,
        use_clip_class_embedding: bool = False,
        clip_model_name: str = "ViT-H-14-378-quickgelu|dfn5b",
        use_proposal_feature: bool = False,
        proposal_type: str = "uni3d-mask3d-gt",
        cuda_device: int = 0,
        normalize_proposal_feature: bool = False,
        use_2d_proposal_feature: bool = False,
        tokenizer: transformers.PreTrainedTokenizer = None,
        only_plans: bool = False,
        sft: bool = False,
        image_encoder: Optional[str] = None,
        image_feature_type: Optional[str] = None,
        use_shared_image_features: bool = True,
        max_image_cache_mb: Optional[int] = None,  # e.g., 10000 for 10GB limit
        prefetch_image_features: bool = True,  # Whether to prefetch
        n_views_in_m_views: str = "32_8",  # e.g., "32_8" means resampling 8 views from 32 views
        add_bracket_in_object_detail: bool = False,
        **kwargs
    ):
        # Initialize base attributes
        self.name = name
        self.data_path = data_path
        self.split = split
        self.ratio = ratio
        self.start_from_last = start_from_last
        self.max_objects = max_objects
        self.max_object_id = max_object_id
        self.seed = seed
        self.add_thinking_trace = add_thinking_trace
        self.add_thinking_trace_prompt = add_thinking_trace_prompt
        self.add_full_thinking_trace_for_filter_in_relational = add_full_thinking_trace_for_filter_in_relational
        self.only_add_positive_relations = only_add_positive_relations
        self.add_plans_first = add_plans_first
        self.fix_template = fix_template
        self.shuffle_objects: bool = shuffle_objects
        self.pre_filter_objects = pre_filter_objects

        self.pc_tokenizer_type = pc_tokenizer_type
        self.object_label_type = object_label_type() if isinstance(object_label_type, type) else object_label_type
        self.frozen_object_type = frozen_object_type
        # self.feature_dim: int = self.object_label_type.num_classes + self.max_object_id + 6

        self.object_classes = self.object_label_type.object_classes

        self.use_clip_class_embedding = use_clip_class_embedding
        self.clip_model_name = clip_model_name
        self.cuda_device = cuda_device

        self.use_proposal_feature = use_proposal_feature
        self.proposal_type = proposal_type
        self.normalize_proposal_feature = normalize_proposal_feature
        self.use_2d_proposal_feature = use_2d_proposal_feature

        self.add_bracket_in_object_detail = add_bracket_in_object_detail

        self.only_plans = only_plans
        self.sft = sft

        self.tokenizer = tokenizer

        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load annotations
        self.annotation_file = self._get_annotation_file(split)
        self.annotations = self._load_annotations()

        self.num_views = int(n_views_in_m_views.split('_')[1]) # e.g., "32_8" -> 8 views
        if image_encoder:
            self.view_annotation = json.load(open(self._get_view_annotation_file(n_views_in_m_views), 'r')) # sample_id -> view annotations
        else:
            self.view_annotation = None
        
        # Load scene data
        self.scene_data = self._load_scene_data()

        self.modality_dims: dict[str, int] = {
            "id": self.max_object_id,
            "location": 6,
        }
        self.modality_order = "3d|id|location"

        # Load object features
        if self.use_clip_class_embedding:
            self.clip_text_embeddings, self.clip_embedding_dim = self._load_clip_embeddings(self.clip_model_name)
            # 更新特征维度，使用CLIP嵌入大小替代one-hot编码
            self.feature_dim = self.clip_embedding_dim + self.max_object_id + 6
            self.class_feature_dim = self.clip_embedding_dim

            self.modality_dims["3d"] = self.clip_embedding_dim
        elif self.use_proposal_feature:
            self.proposal_feature, self.proposal_feature_dim, feature_2d_dim = self._load_proposal_feature(self.proposal_type, normalize=self.normalize_proposal_feature)
            self.feature_dim = self.proposal_feature_dim + self.max_object_id + 6
            self.class_feature_dim = self.proposal_feature_dim

            self.modality_dims["3d"] = self.proposal_feature_dim
            if feature_2d_dim is not None:
                self.modality_dims["3d"] -= feature_2d_dim
                self.modality_dims["2d"] = feature_2d_dim
                self.modality_order = "2d|" + self.modality_order # 2d|3d|id|location
        else:
            self.feature_dim = self.object_label_type.num_classes + self.max_object_id + 6
            self.class_feature_dim = self.object_label_type.num_classes

            self.modality_dims["3d"] = self.object_label_type.num_classes
        
        
        # Prepare object features for each scene
        # self.scene_list = list(self.scene_data.keys())
        # self.scene_list = sorted(list(self.scene_data.keys()))
        self.frozen_features = {}
        self.input_predicted_bboxes = {}
        self.accessed_times = defaultdict(int)
        self._prepare_object_features()

        # Prepare 2D image features if needed
        self.image_encoder_name = image_encoder
        if image_encoder:
            self.image_features = self._load_image_features(
                encoder_name=image_encoder,
                feature_type=image_feature_type,
                use_shared_memory=use_shared_image_features,
                max_cache_mb=max_image_cache_mb,
                prefetch=prefetch_image_features,
            )
        
        # Initialize template formats from Synthetic3DDataset
        self._initialize_templates(**kwargs)
        
        # Generate samples
        # self.samples = self._generate_samples()
        logger.info(f"Received load_from_cache: {load_from_cache}")
        self.samples = self._generate_samples_parallel(load_from_cache) if parallel else self._generate_samples(load_from_cache)
        
        # check uniqueness of scanrefer_id
        scanrefer_ids = [sample["scanrefer_id"] for sample in self.samples]
        if len(scanrefer_ids) != len(set(scanrefer_ids)):
            raise ValueError("scanrefer_id is not unique in the dataset, please check your annotations.")

        # Take partial data if needed
        if ratio < 1.0:
            self._take_partial_data(ratio)

        # Build scanrefer_id to annotation index mapping
        self.scanrefer_id_to_idx = {data["scanrefer_id"]: idx for idx, data in enumerate(self.samples)}
        
        # Log sample examples
        if self.samples:
            logger.info(self.format_sample(self.samples[0]))
            # log a relate_multi and relate_anchor example
            for sample in self.samples:
                if sample["program"] and "relate_multi" in sample["program"]:
                    logger.info(self.format_sample(sample))
                    break
            for sample in self.samples:
                if sample["program"] and "relate_anchor" in sample["program"]:
                    logger.info(self.format_sample(sample))
                    break
        
        if not load_from_cache:
            # self._save_samples()
            pass # NOTE: seems to be too large to save (100K char for each sample)

        # self.stat_word_lengths()

    def _load_image_features(
        self, 
        encoder_name: str, 
        feature_type: str,
        use_shared_memory: bool = True,
        max_cache_mb: Optional[int] = None,
        prefetch: bool = False,
    ):
        """
        Initialize image feature loading using the singleton manager.
        Features are loaded lazily when accessed in __getitem__.
        
        Args:
            encoder_name: Name of the image encoder (for logging)
            feature_type: Type of features ('adaptive_12x12', 'global', or 'patch')
            use_shared_memory: Whether to use shared memory for DDP
            max_cache_mb: Maximum cache size in MB (None for unlimited)
            prefetch: Whether to prefetch all features
        """
        # Get or create the singleton manager
        self.image_feature_manager = ImageFeatureManager(
            use_shared_memory=use_shared_memory,
            max_cache_size_mb=max_cache_mb,
        )
        
        # Register feature paths for all scenes in this dataset
        self.image_feature_manager.register_features(
            data_path=self.data_path,
            feature_type=feature_type,
            scene_ids=self.scene_list
        )
        
        # Optionally prefetch
        if prefetch:
            _rank = dist.get_rank() if dist.is_initialized() else 0
            logger.critical(f"Rank {_rank}: Prefetching image features...")
            self.image_feature_manager.prefetch_scenes(self.scene_list)
            mem_info = self.image_feature_manager.get_memory_usage()
            logger.critical(f"Rank {_rank}: Prefetch complete. "
                    f"Memory usage: {mem_info['total_mb']:.2f} MB")
        
        self.image_feature_type = feature_type
        self.image_encoder_name = encoder_name
        
        rank_info = f" (DDP rank {dist.get_rank()})" if dist.is_initialized() else ""
        logger.info(f"Image feature manager initialized{rank_info} for encoder '{encoder_name}' "
                f"with feature type '{feature_type}'"
                f"{' (shared for DataLoader workers)' if use_shared_memory else ''}")
        

    def exclude_sample_by_scanrefer_id(self, scanrefer_ids: Iterable[str]):
        """Exclude samples by scanrefer_id."""
        scanrefer_ids = set(scanrefer_ids)  # Convert to set for faster lookup
        logger.info(f"Excluding {len(scanrefer_ids)} samples by scanrefer_id.")

        self.samples = [sample for sample in self.samples if sample["scanrefer_id"] not in scanrefer_ids]
        self.scanrefer_id_to_idx = {data["scanrefer_id"]: idx for idx, data in enumerate(self.samples)}
        logger.info(f"Remaining samples: {len(self.samples)}")

    def _load_proposal_feature(self, proposal_type: str, normalize: bool):
        """Load proposal features based on the specified type."""
        proposal_feature_path = self.get_proposal_feature_path(proposal_type)
        logger.info(f"Loading proposal features of type: {proposal_type}, from {proposal_feature_path}")

        if proposal_type in ["uni3d-mask3d-box"]:
            logger.critical("Detected using proposal features. Input boxes would be predicted proposal boxes.")
            self.use_proposal_box_as_input = True
        else:
            self.use_proposal_box_as_input = False

        wanted_keys = ["feature", "mask", "box_corners", "bbox", "object_names"]

        if self.use_2d_proposal_feature:
            wanted_keys += ["feature_2d"]

        proposal_features = torch.load(proposal_feature_path , map_location="cpu")
        proposal_features = {
            # item["scene_id"]: [item["feature"], item["mask"], item["box_corners"]] if "box_corners" in item else [item["feature"], item["mask"]]
            item["scene_id"]: {
                key: item[key] for key in wanted_keys if key in item
            }
            for item in proposal_features if item["scene_id"] in self.scene_list
        }
        # "bbox" keys is raw, may contain more objects than feature, so we remove them
        if "bbox" in next(iter(proposal_features.values())).keys():
            for scene_id in proposal_features.keys():
                proposal_features[scene_id]["bbox"] = proposal_features[scene_id]["bbox"][:len(proposal_features[scene_id]["mask"])]

        logger.info(list(proposal_features.values())[0].keys())
        # Normalize the features to norm=1
        feature_2d_dim = None
        if normalize:
            logger.info("Normalizing proposal features to unit 2-norm.")
            with torch.no_grad():
                for scene_id in proposal_features.keys():
                    feature: torch.Tensor = proposal_features[scene_id]["feature"].to(self.cuda_device) # [N, D]
                    norm = feature.norm(dim=-1, keepdim=True)
                    norm = torch.clamp(norm, min=1e-6)  # Avoid division by zero
                    proposal_features[scene_id]["feature"] = (feature / norm).cpu()

                    if self.use_2d_proposal_feature:
                        try:
                            feature_2d: torch.Tensor = proposal_features[scene_id]["feature_2d"].to(self.cuda_device)
                        except KeyError:
                            logger.warning(f"2D feature not found for scene {scene_id}, keys are: {proposal_features[scene_id].keys()}")
                            raise
                        norm_2d = feature_2d.norm(dim=-1, keepdim=True)
                        norm_2d = torch.clamp(norm, min=1e-6)
                        proposal_features[scene_id]["feature_2d"] = (feature_2d / norm_2d).cpu()

        # concat 2D
        if self.use_2d_proposal_feature:
            logger.info("Using 2D proposal features, concatenating with 3D features.")
            for scene_id in proposal_features.keys():
                feature_2d: torch.Tensor = proposal_features[scene_id]["feature_2d"]

                if feature_2d_dim is None:
                    feature_2d_dim = feature_2d.shape[1]

                feature: torch.Tensor = proposal_features[scene_id]["feature"]
                proposal_features[scene_id]["feature"] = torch.cat([feature_2d, feature], dim=-1)


        proposal_feature_dim = next(iter(proposal_features.values()))["feature"].shape[1]
        logger.info(f"Loaded proposal features with dimension: {proposal_feature_dim}")


        proposal_features = trim_proposal_features(proposal_features, trim_keys=wanted_keys)
        return proposal_features, proposal_feature_dim, feature_2d_dim


    @staticmethod
    def get_proposal_feature_path(type: str="pnpp") -> str:
        if type == "pnpp":
            return f"{DATA_PATH}/scannetv2-pnpp-feature.pkl"
        elif type == "pnpp-vote2cap-box":
            return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature_box_features_281d.pkl" # its box need flip!
            # return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature-new-2_box_features_281d.pkl" # this don't
        elif type == "uni3d-mask3d-box":
            return f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats_200obj2d3d_nms0.975_noinvalid_combined.pt" # 1030d
        elif type == "uni3d-mask3d-gt":
            return f"{SVC_PATH}/pc_features/chatscene_features/scannet_gt_trainval_feat+bbox_feats_200obj2d3d.pt" # 1030d
        elif type == "pnpp-vote2cap-enc":
            return f"{DATA_PATH}/scannetv2-vote2cap-feature_enc_features_259d.pkl"
        else:
            raise ValueError(f"Unknown frozen object feature type: {type}")

    def _load_clip_embeddings(self, model_name="ViT-H-14-378-quickgelu|dfn5b"):
        """Load CLIP model and generate text embeddings for all object classes."""
        import open_clip

        logger.info(f"Loading CLIP model {model_name} for text embeddings...")

        model_name, pretrained = model_name.split("|")

        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        model = model.to(self.cuda_device)
        tokenizer = open_clip.get_tokenizer(model_name)
        
        # Generate text embeddings for all object classes
        logger.info(f"Generating CLIP text embeddings for {len(self.object_classes)} classes...")
        text_embeddings = {}
        with torch.no_grad(), torch.autocast("cuda"):
            for class_name in self.object_classes:
                # Tokenize and encode the class name
                text = f"a {class_name}"  # Adding "a" for better context
                tokens = tokenizer([text])
                tokens = tokens.cuda()
                embedding = model.encode_text(tokens)
                # Normalize the embedding
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                text_embeddings[class_name] = embedding.squeeze().cpu().numpy()
        
        # Get the embedding dimension
        embedding_dim = next(iter(text_embeddings.values())).shape[0]
        logger.info(f"Generated CLIP text embeddings with dimension {embedding_dim}")
        
        # Clean up to free memory
        del model, tokenizer, preprocess
        torch.cuda.empty_cache()
        
        return text_embeddings, embedding_dim

    
    def _get_annotation_file(self, split):
        """Get annotation file path based on dataset name and split."""
        if "sr3d" in self.name.lower():
            files = {
                "train": f"{self.data_path}/sr3d_with_programs_train_enriched.json",
                "val": f"{self.data_path}/sr3d_with_programs_val_enriched.json",
            }
        elif "nr3d" in self.name.lower():
            if "gemini2.5pro" in self.name.lower():
                files = {
                    "train": f"{self.data_path}/nr3d_train_with_program_gemini2.5pro.json",
                    "val": f"{self.data_path}/nr3d_val_with_program_gemini2.5pro.json",
                }
            else: # default Claude Sonnet 3.6
                files = {
                    "train": f"{self.data_path}/nr3d_train_with_program.json",
                    "val": f"{self.data_path}/nr3d_val_with_program.json",
                }

        elif "scanrefer" in self.name.lower():
            files = {
                "train": f"{self.data_path}/ScanRefer_filtered_train_with_program.json",
                "val": f"{self.data_path}/ScanRefer_filtered_val_with_program.json",
            }

        elif "multi3drefer" in self.name.lower():
            files = {
                "train": f"{self.data_path}/multi3drefer/multi3drefer_train.json",
                "val": f"{self.data_path}/multi3drefer/multi3drefer_val.json",
            }

        elif "scannet_attributes" in self.name.lower():
            return []
        else:
            raise ValueError(f"Unknown dataset name: {self.name}")
        
        return files[split]

    def _get_view_annotation_file(self, n_views_in_m_views: str):
        """
        Get view annotation file path based on dataset name and split.
        n_views_in_m_views: e.g., "32_8" => 32 views sampled from 8 viewpoints
        """
        if "sr3d" in self.name.lower():
            return f"{self.data_path}/resampled_views/sr3d_resampled_{n_views_in_m_views}.json"
        elif "nr3d" in self.name.lower():
            return f"{self.data_path}/resampled_views/nr3d_resampled_{n_views_in_m_views}.json"
        elif "scanrefer" in self.name.lower():
            return f"{self.data_path}/resampled_views/scanrefer_resampled_{n_views_in_m_views}.json"
        else:
            # if requires view annotations but dataset is unknown, raise error
            if self.image_encoder_name:
                raise ValueError(f"Unknown dataset name for view annotations: {self.name}")
            else:
                logger.warning(f"No image encoder specified, skipping view annotation file for dataset: {self.name}")
        
        return None
        
    
    def _load_annotations(self):
        """Load annotations from file."""
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        logger.info(f"Loaded {len(annotations)} annotations from {self.annotation_file}")
        return annotations
    
    def _take_partial_data(self, ratio):
        """Take a subset of the data."""
        if self.start_from_last:
            self.samples = self.samples[-int(len(self.samples) * ratio):]
        else:
            self.samples = self.samples[:int(len(self.samples) * ratio)]
        
        logger.info(f"Taking {len(self.samples)} annotations ({ratio:.2f} of total)")
    
    def _generate_samples(self, load_from_cache):
        # NOTE: NOT USED - shall update to match the parallel version
        """Generate training/evaluation samples."""
        if load_from_cache:
            return self._load_samples()

        samples = []

        logger.info(f"Generating samples for {self.name} dataset from programs, may take a while...")
        for idx, anno in enumerate(self.annotations):
            scene_id = anno["scene_id"]
            
            if scene_id not in self.scene_data:
                continue
                
            description = anno["description"]
            program = anno["program"]
            object_id = int(anno.get("object_id"))
            ref_object_ids = anno.get("anchor_ids", [])
            ref_object_classes = anno.get("anchors_types", [])
            ref_object_id_to_class = dict(zip(ref_object_ids, ref_object_classes))
            
            # Get objects for this scene
            objects = self.scene_data[scene_id]["objects"]

            # Create input prompt
            prompt = random.choice(self.instruction_templates).format(description=description)
            
            # Parse program and generate execution trace
            try:
                # Parse program to AST
                ast = parse_program(program, {"objects": objects})
                
                # Execute program to get results and execution trace
                # Pass the target object ID so it's always included in the result
                result, execution_trace = execute_program(ast, {"objects": objects}, object_id, dummy_execution=self.only_plans)
                
                # Generate thinking trace text
                thinking_trace, parts = generate_thinking_trace(execution_trace, {"objects": objects}, description, object_id, ref_object_id_to_class, use_full_detail_for_filter=self.add_full_thinking_trace_for_filter_in_relational, only_add_positive_relations=self.only_add_positive_relations, add_plans_first=self.add_plans_first)
                
                # Generate expected response
                if result:
                    if not self.only_plans: # in only plans, the object results are dummy
                        assert str(result[0]["id"]) == str(object_id), "First object in result must match target object ID"
                    else:
                        # TODO 
                        ...

                    object_details = []
                    for obj in result:
                        location = obj.get("location", [0, 0, 0])
                        size = obj.get("size", [0, 0, 0])
                        object_details.append(
                            random.choice(self.object_detail_templates).format(
                                id=obj["id"],
                                x=location[0],
                                y=location[1],
                                z=location[2],
                                width=size[0],
                                height=size[1],
                                depth=size[2]
                            )
                        )
                    
                    response_body = random.choice(self.response_templates).format(
                        count=len(result),
                        object_details="\n".join(object_details)
                    )

                    if self.add_thinking_trace:
                        expected_response = thinking_trace + "\n" + response_body

                else:
                    logger.warning(f"No objects found for program: {program}")
                    response_body = "Apeiria didn't find any objects matching the description."
            except Exception as e:
                logger.warning(f"Error processing program: {program}, error: {str(e)}")
                print_exc()
                raise e
                continue
            
            target_object = None
            for obj in objects:
                if str(obj["id"]) == str(object_id):
                    target_object = obj
                    break

            if target_object is None:
                logger.warning(f"Target object not found in scene: {scene_id}, object ID: {object_id}. Maybe removed/pre-filtered?")
                continue

            # Create scan2cap_id and scanrefer_id
            scan2cap_id = f"{scene_id}|{object_id}|{target_object['name']}" if object_id is not None else f"{scene_id}|unknown|unknown"
            scanrefer_id = f"{scene_id}|{idx}"
            hash_id = f"real_{scene_id}_{idx}"

            # prompt_words = len(nltk.word_tokenize(prompt))
            # response_words = len(nltk.word_tokenize(expected_response))
            # all_words = prompt_words + response_words
            
            # Store sample
            samples.append({
                # "prompt": prompt,
                # "answer": expected_response,
                "description": prompt,
                "program": program,
                "raw_description": description,
                "scene_id": scene_id,
                "object_id": object_id,
                "object_ids": [object_id],
                "ann_id": anno["ann_id"],
                "result_objects": result,
                "objects": result,  # For reward calculation compatibility
                "question_id": f"{scene_id}_{idx}",
                "raw_question_id": f"{scene_id}_{idx}",
                "scan2cap_id": scan2cap_id,
                "scanrefer_id": scanrefer_id,
                "hash_id": hash_id,
                "data_type": self.name,
                "split": self.split,
                "expected_response": expected_response,
                "thinking_trace_parts": parts,
            })
        
        return samples
    
    def _generate_samples_parallel(self, load_from_cache):
        """Generate training/evaluation samples using multiprocessing."""
        if load_from_cache:
            return self._load_samples()

        # Determine number of processes to use (leave one core free)
        # num_processes = max(1, mp.cpu_count() - 1)
        num_processes = 4
        logger.info(f"Using {num_processes} processes for parallel processing")
        
        # Calculate chunk size - aim for at least 100 items per chunk
        total_annotations = len(self.annotations)
        chunk_size = max(100, total_annotations // (num_processes * 4))
        
        # Split annotations into chunks
        # chunks = [self.annotations[i:i + chunk_size] for i in range(0, total_annotations, chunk_size)]
        chunks = [(self.annotations[i:i + chunk_size], i) for i in range(0, total_annotations, chunk_size)] # include start_idx, to avoid duplicate scanrefer_id

        logger.info(f"Split {total_annotations} annotations into {len(chunks)} chunks")
        
        # Create a partial function with all the shared data
        process_chunk = partial(
            process_annotation_chunk,
            # scene_data=self.scene_data,
            instruction_templates=self.instruction_templates,
            object_detail_templates=self.object_detail_templates,
            response_templates=self.response_templates,
            add_thinking_trace=self.add_thinking_trace,
            name=self.name,
            split=self.split,
            use_full_detail_for_filter=self.add_full_thinking_trace_for_filter_in_relational,
            only_add_positive_relations=self.only_add_positive_relations,
            add_plans_first=self.add_plans_first,
            tokenizer=self.tokenizer,
            dummy_execution=self.only_plans,
            only_plans=self.only_plans,
            is_m3dref=("multi3drefer" in self.name.lower()),
            add_bracket_in_object_detail=self.add_bracket_in_object_detail,
        )
        
        # Process chunks in parallel
        samples = []
        logger.info("Processing annotation chunks in parallel, this may take a while...")
        with mp.Pool(
            processes=num_processes,
            initializer=worker_init,
            initargs=(self.scene_data,)
        ) as pool:
            # Use tqdm to show progress
            results = list(tqdm(
                pool.imap(process_chunk, chunks),
                total=len(chunks),
                desc="Processing annotation chunks"
            ))
            
            # Flatten results
            for chunk_samples in results:
                samples.extend(chunk_samples)

        # sort by ann_id, to make sure the order is consistent
        samples = sorted(samples, key=lambda x: x["scanrefer_id"])
        
        logger.info(f"Generated {len(samples)} samples from {total_annotations} annotations")
        return samples


    def _prepare_object_features_using_proposal_boxes(self, scene_id: str, scene_info: dict):
        """Prepare object features using proposal boxes as input."""
        # note: no frozen_features yet!!!
        num_objects = len(self.proposal_feature[scene_id]["mask"])

        # Create instance_bboxes: [x, y, z, h, w, l, class_id, object_id]
        instance_bboxes = np.zeros((num_objects, 8))

        # Create object features
        object_features = np.zeros((num_objects, self.feature_dim))

        # Create object mask
        object_mask = torch.ones(num_objects, dtype=torch.bool)

        # Create bbox corners
        bbox_corners = np.zeros((num_objects, 8, 3))

        # Calculate IoU matrix between proposal boxes and GT boxes
        # raw_instance_location = scene_info["original_objects"]["locations"] # (N, 3) in raw coord
        raw_instance_location = np.array([obj["location"] for obj in scene_info["original_objects"]]) # (N, 3) in raw coord
        # raw_instance_size = scene_info["original_objects"]["sizes"] # (N, 3) in raw coord
        raw_instance_size = np.array([obj["size"] for obj in scene_info["original_objects"]]) # (N, 3) in raw coord
        raw_instance_bboxes = np.concatenate([raw_instance_location, raw_instance_size], axis=1)

        this_proposal_boxes = self.proposal_feature[scene_id]["bbox"] # (K1, 6)

        # sometimes there is no proposal boxes
        if this_proposal_boxes.shape[0] == 0:
            logger.warning(f"No proposal boxes found for scene {scene_id}, skipping object feature preparation.")
            # TODO: fill with empty data
            self.scene_data[scene_id]["instance_bboxes"] = instance_bboxes
            self.frozen_features[scene_id] = [
                object_features,
                object_mask,
                torch.tensor(bbox_corners, dtype=torch.float32)
            ]
            self.input_predicted_bboxes[scene_id] = instance_bboxes[:, :6]
            return
        elif raw_instance_bboxes.shape[0] == 0:
            logger.warning(f"No GT boxes found for scene {scene_id}, skipping object feature preparation.")
            return

        iou_matrix = mutual_iou_vectorized(this_proposal_boxes, raw_instance_bboxes) # (K1, K2)

        # log mean max iou
        if iou_matrix.size > 0:
            mean_max_iou = iou_matrix.max(axis=1).mean()
            logger.info(f"Mean max Precision IoU in Proposal features in {scene_id}: {mean_max_iou:.4f}") # NOTE: debug to dismiss too much info
            logger.info(f"Mean max Recall IoU in Proposal features in {scene_id}: {iou_matrix.max(axis=0).mean():.4f}")

        if num_objects == 0:
            logger.warning(f"No proposal objects found for scene {scene_id}, skipping.")
            self.scene_data[scene_id]["instance_bboxes"] = instance_bboxes
            self.frozen_features[scene_id] = [
                object_features,
                object_mask,
                torch.tensor(bbox_corners, dtype=torch.float32)
            ]
            self.input_predicted_bboxes[scene_id] = instance_bboxes[:, :6]
            return

        # Fill in data
        for i in range(num_objects):
            proposal_box = self.proposal_feature[scene_id]["bbox"][i]  # [x, y, z, h, w, l]

            # normalize box
            location, size = self._normalize_box(
                proposal_box[:3], proposal_box[3:6],
                ScanNetMixin.GLOBAL_MIN_XYZ, ScanNetMixin.GLOBAL_MAX_XYZ,
                ScanNetMixin.GLOBAL_MIN_SIZE, ScanNetMixin.GLOBAL_MAX_SIZE,
                ScanNetMixin.ROOM_SIZE
            ) # [3], [3]

            location, size = location.tolist(), size.tolist()

            class_name = self.proposal_feature[scene_id]["object_names"][i]
            class_idx = self.object_label_type.name_to_id.get(class_name, -1)
            object_id = i  # Use index as object ID

            # Fill instance_bboxes
            instance_bboxes[i, :3] = location
            instance_bboxes[i, 3:6] = size
            instance_bboxes[i, 6] = class_idx  # Placeholder for class_id
            instance_bboxes[i, 7] = object_id

            # Fill object_features
            object_features[i, :self.class_feature_dim] = self.proposal_feature[scene_id]["feature"][i]

            # ID and location features
            object_features[i, self.class_feature_dim + object_id] = 1.0
            object_features[i, -6:] = np.array(location + size)

            # Generate 8 corners of the box
            for j, (x, y, z) in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]):
                bbox_corners[i, j] = np.array(location) + (np.array([x, y, z]) * np.array(size))


        # Store data
        self.scene_data[scene_id]["instance_bboxes"] = instance_bboxes
        self.frozen_features[scene_id] = [
            object_features,
            object_mask,
            torch.tensor(bbox_corners, dtype=torch.float32)
        ]
        self.input_predicted_bboxes[scene_id] = instance_bboxes[:, :6]
        
        # Add closest_gt_bbox and closest_pred_bbox mappings
        self.scene_data[scene_id]["closest_gt_bbox"] = {}
        self.scene_data[scene_id]["closest_pred_bbox"] = {}
        
        for i in range(num_objects):
            # according to iou_matrix, find the closest gt bbox for each predicted bbox
            matched_gt_idx = iou_matrix[i].argmax().item()
            self.scene_data[scene_id]["closest_gt_bbox"][i] = matched_gt_idx

        for i in range(raw_instance_bboxes.shape[0]):
            # according to iou_matrix, find the closest predicted bbox for each gt bbox
            matched_pred_idx = iou_matrix[:, i].argmax().item()
            self.scene_data[scene_id]["closest_pred_bbox"][i] = matched_pred_idx

    def _prepare_object_features(self):
        """Prepare object features for each scene to match Synthetic3DDataset format."""
        for scene_id, scene_info in self.scene_data.items():
            objects = scene_info["objects"]

            if self.use_proposal_box_as_input:
                # object features, object_mask, bbox_corners are not aligned to GT boxes
                self._prepare_object_features_using_proposal_boxes(scene_id, scene_info)
                continue

            num_objects = len(objects)
            
            # Create instance_bboxes: [x, y, z, h, w, l, class_id, object_id]
            instance_bboxes = np.zeros((num_objects, 8))
            
            # Create object features
            object_features = np.zeros((num_objects, self.feature_dim))
            
            # Create object mask
            object_mask = torch.ones(num_objects, dtype=torch.bool)
            
            # Create bbox corners
            bbox_corners = np.zeros((num_objects, 8, 3))

            
            
            # Fill in data
            for i, obj in enumerate(objects):
                location = obj.get("location", [0, 0, 0])
                size = obj.get("size", [0, 0, 0])
                class_idx = self.object_label_type.name_to_id[obj["name"]]
                object_id = obj["id"]
                if object_id >= self.max_object_id:
                    logger.warning(f"Object ID {object_id} exceeds max_objects limit")
                    object_id = self.max_object_id - 1
                
                # Fill instance_bboxes
                instance_bboxes[i, :3] = location
                instance_bboxes[i, 3:6] = size
                instance_bboxes[i, 6] = class_idx  # Placeholder for class_id
                instance_bboxes[i, 7] = object_id

            # pre-compute box alignment, since the proposal feature is not aligned with the object in order.
            if self.use_proposal_feature:
                # revert normalized box
                raw_location, raw_size = self.get_original_box_from_normalized(
                    instance_bboxes[:, :3], instance_bboxes[:, 3:6], scene_info
                )
                raw_instance_bboxes = np.concatenate([raw_location, raw_size], axis=1)
                this_proposal_boxes = self.proposal_feature[scene_id]["bbox"] # (K1, 6)
                this_proposal_features = self.proposal_feature[scene_id]["feature"] 
                # iou_matrix = mutual_iou(this_proposal_boxes, raw_instance_bboxes) # (K1, K2)
                iou_matrix = mutual_iou_vectorized(this_proposal_boxes, raw_instance_bboxes) # (K1, K2)

                # log mean max iou
                mean_max_iou = iou_matrix.max(axis=1).mean()
                # logger.info(f"Mean max GT IoU in Proposal features in {scene_id}: {mean_max_iou:.4f}") # NOTE: debug to dismiss too much info
            
            # Fill object_features
            #   the order: [object feature -> object id -> location (position+size)]
            for i, obj in enumerate(objects):
                location = obj.get("location", [0, 0, 0])
                size = obj.get("size", [0, 0, 0])
                class_idx = self.object_label_type.name_to_id[obj["name"]]
                object_id = obj["id"]
                if object_id >= self.max_object_id:
                    logger.warning(f"Object ID {object_id} exceeds max_objects limit")
                    object_id = self.max_object_id - 1

                if self.use_clip_class_embedding:
                    # 使用CLIP文本嵌入替代one-hot编码
                    class_name = obj["name"]
                    if class_name in self.clip_text_embeddings:
                        object_features[i, :self.clip_embedding_dim] = self.clip_text_embeddings[class_name]
                    else:
                        logger.warning(f"Class name {class_name} not found in CLIP embeddings")
                elif self.use_proposal_feature:
                    # 使用预训练的特征
                    matched_proposal_idx = iou_matrix[:, i].argmax()
                    matched_iou = iou_matrix[matched_proposal_idx, i]
                    if matched_iou < 0.5:
                        # logger.warning(f"Low IoU ({matched_iou:.4f}) for object {i}, type {obj['name']} in scene {scene_id}")
                        pass # NOTE: debug to dismiss too many such info
                    object_features[i, :self.proposal_feature_dim] = this_proposal_features[matched_proposal_idx]
                else:
                    # 使用原有的one-hot编码
                    object_features[i, class_idx] = 1.0
                
                object_features[i, self.class_feature_dim + object_id] = 1.0
                object_features[i, -6:] = np.array(location + size)
                
                # Generate 8 corners of the box
                for j, (x, y, z) in enumerate([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                                            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]):
                    bbox_corners[i, j] = np.array(location) + (np.array([x, y, z]) - 0.5) * np.array(size)
            
            # Store data
            self.scene_data[scene_id]["instance_bboxes"] = instance_bboxes
            self.frozen_features[scene_id] = [
                object_features,
                object_mask,
                torch.tensor(bbox_corners, dtype=torch.float32)
            ]
            self.input_predicted_bboxes[scene_id] = instance_bboxes[:, :6]
            
            # Add closest_gt_bbox and closest_pred_bbox mappings (identity mapping for real data)
            self.scene_data[scene_id]["closest_gt_bbox"] = {}
            self.scene_data[scene_id]["closest_pred_bbox"] = {}
            
            for i in range(num_objects):
                # Each predicted box maps to itself in GT
                self.scene_data[scene_id]["closest_gt_bbox"][i] = {
                    "gt_id": i,
                    "gt_id_in_array": i,
                    "iou": 1.0  # Perfect IoU with itself
                }
                
                # Each GT box maps to itself in predictions
                self.scene_data[scene_id]["closest_pred_bbox"][i] = {
                    "pred_id": i,
                    "iou": 1.0  # Perfect IoU with itself
                }
    
    def __len__(self):
        return len(self.samples)

    def format_sample(self, sample):
        """Format a sample string for logging."""
        # string = "Example".center(60, "=") + "\n"
        string = f"{self.__class__.__name__} Sample".center(60, "=") + "\n"
        string += f"Scene ID: {sample['scene_id']}\n"
        string += "-" * 60 + "\n"
        string += f"Prompt: {sample['description']}\n"
        string += "-" * 60 + "\n"
        string += f"Expected response: {sample['expected_response']}\n"
        string += "-" * 60 + "\n"
        string += f"Program: {sample['program']}\n"
        string += f"Object ID(s): {sample['object_ids']}\n"
        string += "=" * 60 + "\n"

        return string

    def _save_samples(self):
        # save to data dir
        annotation_file = self.annotation_file.split("/")[-1].replace(".json", "")
        save_file = f"{self.data_path}/{annotation_file}_{self.split}_with_thinking_trace.json"
        # log a sample
        # for key, value in self.samples[0].items():
            # logger.info(f"{key}: {value}, key type: {type(key)}, value type: {type(value)}")
        with locked_file(save_file) as f:
        # with open(save_file, 'w') as f:
            json.dump(self.samples, f, indent=4)

        logger.info(f"Saved {len(self.samples)} samples to {save_file}")

    def _load_samples(self):
        # load from data dir
        annotation_file = self.annotation_file.split("/")[-1].replace(".json", "")
        load_file = f"{self.data_path}/{annotation_file}_{self.split}_with_thinking_trace.json"
        with open(load_file, 'r') as f:
            samples = json.load(f)
        
        logger.info(f"Loaded {len(samples)} samples from {load_file}")
        return samples

    def _get_image_indices(self, data):
        scene_id = data["scene_id"]
        object_id = data["object_id"]
        ann_id = data["ann_id"]

        # NOTE: for Sr3D, ann_id have duplicates under same scene_id and object_id, but their text is extremely similar (synonyms), so we ignore it here.
        view_selection_id = f"{scene_id}-{object_id}-{ann_id}"
        if "sr3d" in self.name.lower():
            view_selection_id = f"{view_selection_id}-{data['raw_description']}" 

        # NOTE: due to an bug, for scenes with N_views < N_sample, we left as blank, so we need to handle it here.
        # check if scene_id in view_annotation exists, if no, take all views under this scene
        scene_id_keys = [key for key in self.view_annotation.keys() if key.startswith(f"{scene_id}-")]
        if len(scene_id_keys) == 0:
            logger.warning(f"View annotation for scene {scene_id} not found, using all available views.")
            image_ids = self.image_feature_manager.get_scene_image_ids(scene_id)
            return image_ids

        image_ids = self.view_annotation[view_selection_id]["sampled_views"]

        return image_ids

    def __getitem__(self, idx):
        """Get a sample in the format expected by Synthetic3DDataset."""
        self.accessed_times[idx] += 1
        
        # Get sample data
        data = self.samples[idx]
        scene_id = data["scene_id"]
        
        # Get object information
        object_id = data["object_id"]
        object_ids = data["object_ids"]
        object_ids = [int(obj_id) for obj_id in object_ids] # Convert to int
        # assert object_id in object_ids, "Target object ID must be in result objects"
        # assert len(object_ids) == 1, "Only one object should be in result objects"
        
        # Get scene data
        # instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()
        # from self.scene_data, get instance_bboxes, since that are always GT boxes
        gt_objects = self.scene_data[scene_id]["objects"]
        gt_boxes = np.array([obj.get("location", [0, 0, 0]) + obj.get("size", [0,0,0]) for obj in gt_objects])
        gt_ids = np.array([obj["id"] for obj in gt_objects])
        gt_class_ids = np.array([self.object_label_type.name_to_id[obj["name"]] for obj in gt_objects])
        
        instance_bboxes = np.zeros((len(gt_objects), 8))
        instance_bboxes[:, :6] = gt_boxes
        instance_bboxes[:, 6] = gt_class_ids
        instance_bboxes[:, 7] = gt_ids
        
        # Set target_id and target_bbox
        target_ids = [] # target_id is the index in instance_bboxes
        all_object_ids = instance_bboxes[:, 7].astype(int)
        for _object_id in object_ids:
            for i, obj_id in enumerate(all_object_ids):
                if str(obj_id) == str(_object_id):
                    target_ids.append(i)

        if len(target_ids) == 0:
            if  len(object_ids) > 0:
                if self.__class__.__name__ == "Real3DDenseCaptioningDataset" and self.split != "train":
                    pass # using proposal box as sample, target_id and object_ids don't correspond.
                else:
                    logger.warning(f"Object IDs {object_ids} not found in scene {scene_id}")

            target_id = 0
        else:
            target_id = target_ids[0]  # Take the first one as target_id

        target_bbox = instance_bboxes[target_id, 0:6].copy() if target_id < len(instance_bboxes) else np.zeros(6)
        
        # Get target predicted ID
        target_pred_id = target_id  # For real data, pred_id = id # NOTE: change for proposal box input
        
        # Get object features
        object_feature = self.frozen_features[scene_id][0]
        object_mask = self.frozen_features[scene_id][1]
        predicted_bbox_corners = self.frozen_features[scene_id][2]
        input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        
        # Get object labels and IOUs
        object_labels = instance_bboxes[:, 6].astype(np.int64)
        object_ious = np.ones(len(object_feature), dtype=np.float32)  # All 1.0 for real data

        # Compose an list of target object info, for reward calculation
        target_object_info = []
        for _target_id in target_ids:
            target_object_info.append({
                "id": int(instance_bboxes[_target_id, 7]),
                "location": target_bbox[:3].tolist(),
                "position": target_bbox[:3].tolist(),
                "size": target_bbox[3:6].tolist(),
            })

        # Apply shuffling if needed
        if self.shuffle_objects and self.split == "train":
            generator = np.random.default_rng(seed=idx + self.accessed_times[idx] + self.seed)
            shuffle_indices = generator.permutation(len(object_feature))
            revert_indices = np.argsort(shuffle_indices)
            
            object_feature = object_feature[shuffle_indices]
            object_mask = object_mask[shuffle_indices]
            predicted_bbox_corners = predicted_bbox_corners[shuffle_indices]
            input_predicted_bbox = input_predicted_bbox[shuffle_indices]
            object_labels = object_labels[shuffle_indices]
            object_ious = object_ious[shuffle_indices]
            
            # Update target_pred_id after shuffling
            target_pred_id = revert_indices[target_pred_id].item()
        else:
            shuffle_indices = np.arange(len(object_feature))
            revert_indices = np.arange(len(object_feature))
        
        # Create PC dictionary
        pc_dict = {
            "object_feature": object_feature,
            "object_mask": object_mask,
            "predicted_bbox_corners": predicted_bbox_corners, # actually not used in training
            "input_predicted_bbox": input_predicted_bbox,
            "object_labels": object_labels,
            "object_ious": object_ious,
        }

        # load 2D features if needed
        if self.image_encoder_name:
            image_ids = self._get_image_indices(data)
            pc_dict["image_embeds"] = self._get_image_features(scene_id, image_ids) # [N_views, L, D] or [N_views, 1, D]

        # calculate a id->box map for both proposal and GT boxes
        # so that during training, we can easily get the box by its id
        proposal_id2box = {}
        for i in range(len(self.input_predicted_bboxes[scene_id])):
            # for proposal, index is the id (since no id)
            proposal_id2box[str(i)] = self.input_predicted_bboxes[scene_id][i].tolist()

        gt_id2box = {}
        for i, obj in enumerate(gt_objects):
            gt_id2box[str(obj["id"])] = gt_boxes[i].tolist()
        
        return {
            # 2D instruction, image, target
            "question_id": data["question_id"],
            "raw_question_id": data["raw_question_id"],
            **({
                "raw_description": data["raw_description"], # NOTE: comment when pre-training
                "prompt_with_plan": data["prompt_with_plan"], # NOTE: comment when pre-training
            } if not self.sft else {}),
            "scan2cap_id": data["scan2cap_id"],
            "scene_id": scene_id,
            "scanrefer_id": data["scanrefer_id"],
            "hash_id": data["hash_id"],
            "target_id": target_id,  # index in GT bboxes
            "target_pred_id": target_pred_id,  # index in predicted bboxes
            "object_id": object_id,  # index in GT bbox index
            "object_ids": object_ids,  # all result objects
            "objects": target_object_info,
            "data_type": self.name,
            "split": self.split,
            "target_bbox": target_bbox,
            "program": data.get("program", ""),
            "program_complexity": self.count_program_complexity(data.get("program", "")),
            "description": data["description"],
            "expected_response": data["expected_response"],
            "shuffle_indices": shuffle_indices,
            "revert_indices": revert_indices,
            "proposal_id2box": proposal_id2box,
            "gt_id2box": gt_id2box,
            # 3D
            **pc_dict,
        }

    def stat_word_lengths(self):
        """Calculate word lengths for prompt and response."""
        prompt_lengths = []
        response_lengths = []
        for sample in self.samples:
            prompt_lengths.append(sample["prompt_words"])
            response_lengths.append(sample["response_words"])
        
        prompt_lengths = np.array(prompt_lengths)
        response_lengths = np.array(response_lengths)
        all_lengths = prompt_lengths + response_lengths
        
        logger.info(f"Prompt: mean={prompt_lengths.mean():.2f}, std={prompt_lengths.std():.2f}, max={prompt_lengths.max()}")
        logger.info(f"Response: mean={response_lengths.mean():.2f}, std={response_lengths.std():.2f}, max={response_lengths.max()}")
        logger.info(f"All: mean={all_lengths.mean():.2f}, std={all_lengths.std():.2f}, max={all_lengths.max()}")
    
    @staticmethod
    def count_program_complexity(program: str) -> int:
        """Count the number of steps in the program, by counting parentheses"""
        simple_functions = ["scene", "intersection", "intersect", "union", "intersect", "exclude"]
        parentheses = program.count("(")
        for func in simple_functions:
            parentheses -= program.count(f"{func}(")
        return parentheses

    @property
    def is_m3dref(self):
        return "multi3drefer" in self.name.lower()
    
    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False, iou_thresholds=[0.25,0.5]) -> Tuple[str, Dict]:
        """
        Evaluate predictions against ground truth.
        Reuses the evaluation logic from Synthetic3DDataset.
        """
        correct_iou = 0
        correct_class = 0
        # correct_ids = 0
        # total = 0
        ious = []
        
        # Metrics for precision and recall
        # total_pred_boxes = 0
        # total_gt_boxes = 0
        # true_positives = 0  # Predicted boxes that match a GT box
        # detected_gt_boxes = 0  # GT boxes that are detected by any predicted box

        # record per-sample metrics
        # iou_threshold -> list of metrics
        recalls, precisions, f1_scores = defaultdict(list), defaultdict(list), defaultdict(list)
        id_accuracies = []
        
        # Process predictions to extract object ID(s)
        processed_preds = {}
        for key, pred in preds.items():
            parsed_objects = parse_response(pred)
            # logger.info(f"{key}: parsed objects: {parsed_objects}")
            if parsed_objects is None:
                logger.warning(f"No objects found in {pred}")
            processed_preds[key] = parsed_objects
        
        # Evaluate predictions
        common_keys = set(processed_preds.keys()) & set(gt_indices.keys())
        logger.info(f"Common keys: {len(common_keys)}")
        logger.info(f"Total predictions: {len(processed_preds)}")
        logger.info(f"Total GT: {len(gt_indices)}")
        
        for key in common_keys:
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id if needed
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            scene_id = scanrefer_id.split("|")[0]
            ann_idx = int(scanrefer_id.split("|")[1])
            
            # Get sample data
            sample = None
            if scanrefer_id in self.scanrefer_id_to_idx:
                sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            if sample is None:
                logger.warning(f"Length of scanrefer_id_to_idx: {len(self.scanrefer_id_to_idx)}")
                ic(scanrefer_id, self.scanrefer_id_to_idx)
                logger.warning(f"Sample not found for {scanrefer_id}")
                continue
            
            # Get predicted and ground truth object ID(s)
            pred_ids = [obj["id"] for obj in processed_preds[key]]
            gt_ids = sample["object_ids"]
            
            # TODO: for zero-target (no object) in gt, recall is always 1, precision is always 0 if any pred_ids exist, or 1 if no pred_ids.
            if len(gt_ids) == 0:
                for thr in iou_thresholds:
                    recalls[thr].append(1.0)
                    if len(pred_ids) == 0:
                        precisions[thr].append(1.0)
                    else:
                        precisions[thr].append(0.0)
                    f1_scores[thr].append(0.0 if len(pred_ids) > 0 else 1.0)

                id_accuracies.append(1.0 if len(pred_ids) == 0 else 0.0)
                continue
            
            # Count total boxes for precision/recall
            # total_pred_boxes += len(pred_ids)
            # total_gt_boxes += len(gt_ids)
            
            # Check if at least one predicted ID matches a ground truth ID
            # has_match = any(pred_id in gt_ids for pred_id in pred_ids)
            has_match = set(pred_ids) & set(gt_ids)
            has_match = len(has_match) > 0
            

            id_accuracies.append(1.0 if has_match else 0.0)
            
            # Evaluate IoU for all predicted boxes against all GT boxes
            pred_boxes = []
            for obj in processed_preds[key]:
                pred_boxes.append([obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]])
            pred_boxes = np.array(pred_boxes)
            
            gt_boxes = []
            for gt_id in gt_ids:
                objects = self.scene_data[scene_id]["objects"]
                for obj in objects:
                    if obj["id"] == gt_id:
                        location = obj.get("location", [0, 0, 0])
                        size = obj.get("size", [0, 0, 0])
                        gt_boxes.append(location + size)
                        break
            assert len(gt_boxes) == len(gt_ids), f"GT boxes length mismatch: {len(gt_boxes)} vs {len(gt_ids)}"
            gt_boxes = np.array(gt_boxes)
            
            # Calculate IoU matrix between all pred_boxes and gt_boxes
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = box3d_iou_orthogonal(pred_box, gt_box)

            # Calculate mean IoU for this sample
            if len(iou_matrix) > 0:
                # For each predicted box, find its max IoU with any GT box
                max_ious = iou_matrix.max(axis=1) if len(gt_boxes) > 0 else np.zeros(len(pred_boxes))
                ious.extend(max_ious.tolist())
            
            if self.is_m3dref:
                # Use Hungarian algorithm to find best matching
                # pad to square matrix
                max_dim = max(iou_matrix.shape)
                padded_iou_matrix = np.zeros((max_dim, max_dim))
                padded_iou_matrix[:iou_matrix.shape[0], :iou_matrix.shape[1]] = iou_matrix

                row_idx, col_idx = linear_sum_assignment(-padded_iou_matrix)  # maximize IoU
               
                for thr in iou_thresholds:
                    _tp = 0
                    for i in range(len(pred_ids)):
                        iou = padded_iou_matrix[row_idx[i], col_idx[i]]
                        if iou >= thr:
                            _tp += 1  # True positive
                    precisions[thr].append(_tp / len(pred_ids) if len(pred_ids) > 0 else 0)
                    recalls[thr].append(_tp / len(gt_ids) if len(gt_ids) > 0 else 0)
                    f1_scores[thr].append(2 * _tp / (len(pred_ids) + len(gt_ids)) if (len(pred_ids) + len(gt_ids)) > 0 else 0)

                
            else:
                # Calculate precision and recall at different IoU thresholds
                for thr in iou_thresholds:
                    # For precision: check if each predicted box has IoU > threshold with any GT box
                    pred_matches = (iou_matrix.max(axis=1) > thr).sum() if len(gt_boxes) > 0 else 0
                    
                    # For recall: check if each GT box has IoU > threshold with any predicted box
                    gt_matches = (iou_matrix.max(axis=0) > thr).sum() if len(pred_boxes) > 0 else 0
                

                    # Calculate per-sample precision, recall, F1
                    sample_precision = pred_matches / len(pred_ids) if len(pred_ids) > 0 else 0
                    sample_recall = gt_matches / len(gt_ids) if len(gt_ids) > 0 else 0
                    sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0

                    precisions[thr].append(sample_precision)
                    recalls[thr].append(sample_recall)
                    f1_scores[thr].append(sample_f1)
            
        
        # Calculate metrics
        ious = np.array(ious)
        id_accuracy = np.mean(id_accuracies) if len(id_accuracies) > 0 else 0
        mean_iou = np.mean(ious) if len(ious) > 0 else 0
        total = len(common_keys)
        
        # Calculate precision, recall, and F1
        precisions = {thr: np.mean(precisions[thr]) if len(precisions[thr]) > 0 else 0 for thr in iou_thresholds}
        recalls = {thr: np.mean(recalls[thr]) if len(recalls[thr]) > 0 else 0 for thr in iou_thresholds}
        f1_scores = {thr: np.mean(f1_scores[thr]) if len(f1_scores[thr]) > 0 else 0 for thr in iou_thresholds}

        
        # Create evaluation message
        message = (
            "Object Identification Task:\n"
            f"[Acc@{iou_threshold:.2f}] "
            f"Mean IoU: {mean_iou:.4f}, "
            f"ID: {id_accuracy:.4f}, "
            f"Total: {total}"
        )
        
        metrics = {
            "id_accuracy": id_accuracy,
            "mean_iou": mean_iou.item() if isinstance(mean_iou, np.ndarray) else mean_iou,
            "precision": precisions[iou_threshold], # compatibility to old evaluation
            "recall": recalls[iou_threshold],
            "f1_score": f1_scores[iou_threshold],
            "total": total,
        }

        for thr in iou_thresholds:
            message += (
                f"\nPrecision@{thr:.2f}: {precisions[thr]:.4f}, "
                f"Recall@{thr:.2f}: {recalls[thr]:.4f}, "
                f"F1@{thr:.2f}: {f1_scores[thr]:.4f}"
            )
            metrics.update({
                f"precision_at_{thr}": precisions[thr],
                f"recall_at_{thr}": recalls[thr],
                f"f1_score_at_{thr}": f1_scores[thr],
            })
        
        return message, metrics
    
 
    
    def get_all_scene_ids(self):
        """Get all scene ID(s) in the dataset"""
        return self.scene_list
    
    def get_dataset_description(self):
        """Get a description of the dataset"""
        return f"{self.__class__.__name__}-{self.name}-{self.split}"

class Real3DObjectInfoDataset(Real3DDataset):
    """
    A simplified version of the Synthetic3DDataset that focuses on a single task:
    For each object in each scene, output its ID, class, location, and size.
    """
    
    def __init__(
        self,
        name: str = "sr3d_object_info",
        no_object_id_input: bool = False,
        **kwargs
    ):
        # Override prompt templates
        kwargs["instruction_templates"] = kwargs.get("instruction_templates", None) or ([
            "Describe the object with ID {object_id} in the scene. Provide its class, position, and dimensions.\n",
            "For object {object_id} in the scene, specify its class, location, and size.\n",
            "Tell me about object {object_id} in the scene. What is its class, position, and dimensions?\n",
            "Provide details for object {object_id} in the scene. Include category, position, and size.\n",
            "What can you tell me about object {object_id} in the scene? Its category, location, and dimensions, please.\n"
        ] if not no_object_id_input else [
            "Describe the object in the scene. Provide its class, position, and dimensions.\n",
            "For the object in the scene, specify its class, location, and size.\n",
            "Tell me about the object in the scene. What is its class, position, and dimensions?\n",
            "Provide details for the object in the scene. Include category, position, and size.\n",
            "What can you tell me about the object in the scene? Its category, location, and dimensions, please.\n"
        ])

        
        # Override response templates to work with object_detail_templates
        kwargs["response_templates"] = kwargs.get("response_templates", None) or [
            "Roger. The object is {object_class}. {object_details}",
            "Apeiria found {object_class}: {object_details}",
            "Object {object_class} details: {object_details}",
            "Here are the details for {object_class}: {object_details}",
            "Apeiria has analyzed the {object_class}: {object_details}"
        ]

        kwargs["add_thinking_trace"] = False # for single object, no need to add thinking trace
        kwargs["parallel"] = False  # Disable parallel processing for now

        # Initialize with parent class but override name
        super().__init__(name=name, **kwargs)

    def _generate_samples(self, load_from_cache):
        """
        Generate data for the object information task.
        For each object in each scene, create a single annotation that asks about that object.
        """
        logger.info(f"Generating object info annotations from {len(self.scene_list)} scene layouts...")
        
        samples = []

        # Create one annotation per object in each scene
        for scene_id in self.scene_list:
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)
            real_object_indices = instance_bboxes[:, 7].astype(int)
            
            # For each object in the scene
            for obj_idx in range(len(object_classes)):
                class_name = self.object_classes[int(object_classes[obj_idx])]
                
                # Create a description/question asking about this specific object
                description = random.choice(self.instruction_templates).format(object_id=obj_idx)
                
                # Create the object detail using the existing templates
                obj_detail = random.choice(self.object_detail_templates).format(
                    id=real_object_indices[obj_idx],
                    object_class=class_name,
                    x=locations[obj_idx, 0],
                    y=locations[obj_idx, 1],
                    z=locations[obj_idx, 2],
                    width=locations[obj_idx, 3],
                    height=locations[obj_idx, 4],
                    depth=locations[obj_idx, 5]
                )
                
                # Create the expected response
                expected_response = random.choice(self.response_templates).format(
                    object_id=real_object_indices[obj_idx],
                    object_class=class_name,
                    object_details=obj_detail
                )
                
                # Add annotation
                ann_id = len(samples)
                scanrefer_id = f"{scene_id}|{ann_id}"
                hash_id = f"real_objinfo_{scene_id}_{ann_id}"
                scan2cap_id = f"{scene_id}|{real_object_indices[obj_idx]}|{class_name}"
                question_id = f"{scene_id}_{ann_id}"
                raw_question_id = f"{scene_id}_{ann_id}"
                
                samples.append({
                    "scene_id": scene_id,
                    "ann_id": str(ann_id),
                    "description": description,
                    "program": f"get_object(scene(), {real_object_indices[obj_idx]})",  # Simple program to get this object
                    "object_name": class_name,
                    "object_id": real_object_indices[obj_idx],  # This specific object
                    "object_ids": [real_object_indices[obj_idx]],  # Just this one object
                    "scanrefer_id": scanrefer_id,
                    "hash_id": hash_id,
                    "expected_response": expected_response,
                    "question_id": question_id,
                    "raw_question_id": raw_question_id,
                    "scan2cap_id": scan2cap_id,
                })
        
        logger.info(f"Generated {len(samples)} object info annotations across {len(self.scene_list)} scenes")
        return samples
    
    def __getitem__(self, idx):
        """
        Override the __getitem__ method to focus on a single object.
        """
        self.accessed_times[idx] += 1
        
        # Get annotation data
        data = self.samples[idx]
        scene_id = data["scene_id"]
        ann_id = data["ann_id"]
        question_id = data["question_id"]
        raw_question_id = data["raw_question_id"]
        scan2cap_id = data["scan2cap_id"]
        scanrefer_id = data["scanrefer_id"]
        hash_id = data["hash_id"]
        description = data["description"]
        program = data["program"]
        program_complexity = 1  # Simple program
        
        # Get object information - this is for a single object
        object_name = data["object_name"]
        object_id = data["object_id"]
        object_ids = data["object_ids"]  # Should be a list with just one ID
        
        # Get scene data
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()
        target_id = None # the index in the instance_bboxes array
        all_object_ids = instance_bboxes[:, 7].astype(int)
        for i, obj_id in enumerate(all_object_ids):
            if str(obj_id) == str(object_id):
                target_id = i
                break
        if target_id is None:
            logger.warning(f"Object ID {object_id} not found in scene {scene_id}")
            target_id = 0

        target_bbox = instance_bboxes[target_id, 0:6].copy()
        
        # Get target predicted ID (same as object_id for synthetic data)
        # FIXME: is it correct?
        target_pred_id = object_id
        
        # Get object features - but only for this specific object
        all_object_features = self.frozen_features[scene_id][0]
        all_object_mask = self.frozen_features[scene_id][1]
        all_predicted_bbox_corners = self.frozen_features[scene_id][2]
        all_input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        all_object_labels = instance_bboxes[:, 6].astype(np.int64)
        
        # Create a mask that only includes this object
        single_object_mask = torch.zeros_like(all_object_mask)
        single_object_mask[target_id] = 1
        
        # Create PC dictionary with just this object
        pc_dict = {
            "object_feature": all_object_features,  # Keep all features but use mask
            "object_mask": single_object_mask,      # Only this object is visible
            "predicted_bbox_corners": all_predicted_bbox_corners,
            "input_predicted_bbox": all_input_predicted_bbox,
            "object_labels": all_object_labels,
            "object_ious": np.ones(len(all_object_features), dtype=np.float32),  # All 1.0 for synthetic data
        }

        if self.image_encoder_name:
            image_ids = self._get_image_indices(data)
            pc_dict["image_embeds"] = self._get_image_features(scene_id, image_ids) # [N_views, L, D] or [N_views, 1, D]
        
        return {
            # 2D instruction, image, target
            "question_id": question_id,
            "raw_question_id": raw_question_id,
            "scan2cap_id": scan2cap_id,
            "scene_id": scene_id,
            "scanrefer_id": scanrefer_id,
            "hash_id": hash_id,
            "target_id": target_id,  # index in GT bboxes
            "target_pred_id": target_pred_id,  # index in predicted bboxes
            "object_id": object_id,  # index in GT bbox index
            "object_ids": object_ids,  # just this one object
            "data_type": self.name,
            "split": self.split,
            "target_bbox": target_bbox,
            "program": program,
            "program_complexity": program_complexity,
            "description": description,
            "expected_response": data["expected_response"],
            "shuffle_indices": np.arange(len(all_object_features)),  # No shuffling needed
            "revert_indices": np.arange(len(all_object_features)),
            # 3D
            **pc_dict,
        }

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)
    
    def evaluate(self, preds, gt_indices, iou_threshold=0.25, hash_id_index: bool=False, use_closest_gt: bool=False) -> Tuple[str, Dict]:
        """
        Evaluate predictions for the single object information task.
        
        Args:
            preds: Dict mapping scanrefer_id to predicted response text
            gt_indices: Dict mapping scanrefer_id to GT object indices (single object per sample)
            iou_threshold: IoU threshold for considering a prediction correct
            hash_id_index: Whether keys in preds are hash_ids instead of scanrefer_ids
            use_closest_gt: Whether to use closest GT bbox for evaluation
            
        Returns:
            message: Evaluation message
            metrics: Dictionary of evaluation metrics
        """
        total_objects = 0
        correctly_identified_objects = 0
        correctly_classified_objects = 0
        position_errors = []
        dimension_errors = []
        ious = []

        # Process predictions to extract object information
        for key, pred in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id if needed
                for ann in self.annotation:
                    if ann["hash_id"] == key:
                        scanrefer_id = ann["scanrefer_id"]
                        break
            
            # Find the corresponding annotation
            sample = None
            if scanrefer_id in self.scanrefer_id_to_idx:
                sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            if sample is None:
                logger.warning(f"Length of scanrefer_id_to_idx: {len(self.scanrefer_id_to_idx)}")
                ic(scanrefer_id, self.scanrefer_id_to_idx)
                logger.warning(f"Sample not found for {scanrefer_id}")
                continue
            
            scene_id = sample["scene_id"]
            object_id = sample["object_id"]
            
            # Get ground truth data for this object
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            target_id = None
            for i, obj_id in enumerate(instance_bboxes[:, 7].astype(int)):
                if str(obj_id) == str(object_id):
                    target_id = i
                    break

            gt_class = self.object_classes[int(instance_bboxes[target_id, 6])]
            gt_position = instance_bboxes[target_id, :3]
            gt_dimensions = instance_bboxes[target_id, 3:6]
            gt_bbox = instance_bboxes[target_id, :6]
            
            # Parse the response to extract object information
            parsed_objects = parse_response(pred)
            # logger.info(f"Pred: {pred}, Parsed: {parsed_objects}")
            
            total_objects += 1
            
            # Check if the object was correctly identified
            for obj in parsed_objects:
                if obj["id"] == object_id:
                    correctly_identified_objects += 1
                    
                    # Check if class is mentioned in the response
                    if gt_class.lower() in pred.lower():
                        correctly_classified_objects += 1
                    
                    # Calculate position and dimension errors
                    pred_position = np.array([obj["x"], obj["y"], obj["z"]])
                    pred_dimensions = np.array([obj["width"], obj["height"], obj["depth"]])
                    
                    position_error = np.linalg.norm(pred_position - gt_position)
                    dimension_error = np.linalg.norm(pred_dimensions - gt_dimensions)
                    
                    position_errors.append(position_error)
                    dimension_errors.append(dimension_error)

                    # Calculate IoU with GT bbox
                    pred_bbox = np.array([obj["x"], obj["y"], obj["z"], obj["width"], obj["height"], obj["depth"]])
                    iou = box3d_iou_orthogonal(pred_bbox, gt_bbox)
                    ious.append(iou)
                    
                    break
        
        # Calculate metrics
        #   handle numpy float to Python float to let visualization clean
        identification_accuracy = correctly_identified_objects / total_objects if total_objects > 0 else 0
        classification_accuracy = correctly_classified_objects / total_objects if total_objects > 0 else 0
        
        mean_position_error = np.mean(position_errors).tolist() if position_errors else float('inf')
        mean_dimension_error = np.mean(dimension_errors).tolist() if dimension_errors else float('inf')

        ious = np.array(ious)
        mean_iou = np.mean(ious).tolist() if len(ious) > 0 else 0
        iou_accuracy = ((ious > iou_threshold).sum() / total_objects).tolist() if total_objects > 0 else 0
        
        # Create evaluation message
        message = (
            f"Single Object Info Task:\n"
            f"ID Acc: {identification_accuracy:.4f}, "
            f"Classification Acc: {classification_accuracy:.4f}, "
            f"Mean Position Error: {mean_position_error:.4f}, "
            f"Mean Dimension Error: {mean_dimension_error:.4f}, "
            f"Mean IoU: {mean_iou:.4f}, "
            f"IoU Acc@{iou_threshold:.2f}: {iou_accuracy:.4f}, "
            f"Total Objects: {total_objects}"
        )
        
        metrics = {
            "identification_accuracy": identification_accuracy,
            "classification_accuracy": classification_accuracy,
            "mean_position_error": mean_position_error,
            "mean_dimension_error": mean_dimension_error,
            "mean_iou": mean_iou,
            "iou_accuracy": iou_accuracy,
            "total_objects": total_objects,
            "correctly_identified_objects": correctly_identified_objects,
            "correctly_classified_objects": correctly_classified_objects,
        }
        
        return message, metrics
    
    def _save_samples(self):
        pass

    def _load_samples(self):
        pass

class Real3DFilterDataset(Real3DDataset):
    def __init__(
        self,
        name: str = "sr3d_filter",
        add_partial_full_thinking_trace_for_filter: bool = False,
        max_filter_objects: int = 1000, # no limit
        **kwargs
    ):
        # Override prompt templates
        kwargs["instruction_templates"] = kwargs.get("instruction_templates") or [
            "Identify all {class_name}s in the scene and provide their IDs, locations, and sizes.\n",
            "Find all {class_name}s in this scene. For each one, provide its ID, position, and dimensions.\n",
            "List all {class_name}s with their IDs, coordinates, and sizes.\n",
            "Where are all the {class_name}s in this scene? Give me their IDs, positions, and dimensions.\n",
            "Locate all {class_name}s in the scene. For each one, specify its ID, location, and size.\n"
        ]

        
        # Override response templates to work with object_detail_templates
        kwargs["response_templates"] = kwargs.get("response_templates") or [
            "Apeiria found {count} {class_name}(s) in the scene:\n{object_details}",
            "Roger. There are {count} {class_name}(s) in this scene:\n{object_details}",
            "Roger. The scene contains {count} {class_name}(s):\n{object_details}",
            "{count} {class_name}(s) identified:\n{object_details}",
            "Apeiria has located {count} {class_name}(s):\n{object_details}"
        ]

        kwargs["thinking_trace_template"] = kwargs.get("thinking_trace_template") or [
            "[APEIRIA THINKS]\n"
            "Apeiria will now analyze the scene and identify the requested object.\n"
            "First, let me list all {object_count} objects and their details:\n"
            "{object_details_with_class}\n"
            "Now, Apeiria need to identify all {class_name}(s) in the scene. "
            "According to the above analyzed object details, those objects are:\n"
            "Object {object_ids_with_class}\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]

        kwargs["parallel"] = False

        self.add_partial_full_thinking_trace_for_filter = add_partial_full_thinking_trace_for_filter
        self.max_filter_objects = max_filter_objects

        # Initialize with parent class but override name
        super().__init__(name=name, **kwargs)

    def _generate_samples(self, load_from_cache):
        """Generate data (annotations, questions, expected responses) from scene layouts"""
        logger.info(f"Generating annotations from {len(self.scene_list)} scene layouts...")
        
        samples = []
        # Create annotations for each class in each scene
        for scene_id in self.scene_list:
            # objects_per_class = self.scene_data[scene_id].get("objects_per_class", {})
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)
            real_object_indices = instance_bboxes[:, 7].astype(int) # the assigned fixed ID
            num_objects = len(object_classes)

            # find classes in the scene
            classes_in_scene = set(object_classes.tolist())
            
            for class_idx in classes_in_scene:
                class_name = self.object_classes[class_idx]
                
                # Find all objects of this class
                object_indices = [i for i, c in enumerate(object_classes) if c == class_idx]
                
                if not object_indices:
                    continue
                
                # Create a program that filters objects of this class
                program = f"filter(scene(), {class_name})"
                
                # Create a description/question
                description = random.choice(self.instruction_templates).format(class_name=class_name)
                
                # Create the expected response
                object_details = []
                for i in object_indices:
                    obj_detail = random.choice(self.object_detail_templates).format(
                        id=real_object_indices[i],
                        x=locations[i, 0],
                        y=locations[i, 1],
                        z=locations[i, 2],
                        width=locations[i, 3],
                        height=locations[i, 4],
                        depth=locations[i, 5]
                    )
                    object_details.append(obj_detail)
                
                object_details_str = "\n".join(object_details)
                
                response_template = random.choice(self.response_templates)
                
                expected_response = response_template.format(
                    count=len(object_indices),
                    class_name=class_name,
                    object_details=object_details_str
                )

                # Add thinking trace if enabled
                if self.add_thinking_trace:
                    if len(object_indices) > self.max_filter_objects:
                        # too long the thinking trace, skip
                        continue 
                    # Create detailed object descriptions with class names for all objects
                    object_details_with_class = []
                    if self.add_partial_full_thinking_trace_for_filter:
                        # use only object ID and class name
                        for i in range(len(object_classes)):
                            obj_name = self.object_classes[int(object_classes[i])]
                            obj_id_with_class = self.object_id_with_class_templates.format(
                                id=real_object_indices[i],
                                object_name=obj_name
                            )
                            object_details_with_class.append(obj_id_with_class)
                        
                        object_details_with_class_str = ", ".join(object_details_with_class)
                    else:
                        for i in range(len(object_classes)):
                            obj_name = self.object_classes[int(object_classes[i])]
                            obj_detail = random.choice(self.object_detail_with_class_templates).format(
                                id=real_object_indices[i],
                                object_name=obj_name,
                                x=locations[i, 0],
                                y=locations[i, 1],
                                z=locations[i, 2],
                                width=locations[i, 3],
                                height=locations[i, 4],
                                depth=locations[i, 5]
                            )
                            object_details_with_class.append(obj_detail)

                        object_details_with_class_str = "\n".join(object_details_with_class)
                    
                    # Create list of objects of the target class
                    object_ids_with_class = []
                    for i in object_indices:
                        obj_id_with_class = self.object_id_with_class_templates.format(
                            id=real_object_indices[i],
                            object_name=class_name
                        )
                        object_ids_with_class.append(obj_id_with_class)
                    
                    # Format the thinking trace
                    # ic(self.thinking_trace_template)
                    thinking_trace = random.choice(self.thinking_trace_template).format(
                        object_count=num_objects,
                        object_details_with_class=object_details_with_class_str,
                        class_name=class_name,
                        object_ids_with_class=", ".join(object_ids_with_class)
                    )
                    
                    # Combine thinking trace with expected response
                    expected_response = thinking_trace + expected_response
                    
                
                # Add annotation
                ann_id = len(samples)
                scanrefer_id = f"{scene_id}|{ann_id}"
                hash_id = f"real_filter_{scene_id}_{ann_id}"
                scan2cap_id = f"{scene_id}|{class_name}"
                question_id = f"{scene_id}_{ann_id}"
                raw_question_id = f"{scene_id}_{ann_id}"
                
                samples.append({
                    "scene_id": scene_id,
                    "ann_id": str(ann_id),
                    "description": description,
                    "program": program,
                    "object_name": class_name,
                    "object_ids": real_object_indices[object_indices].tolist(),
                    "object_id": real_object_indices[object_indices].tolist()[0],  # First object ID
                    "scanrefer_id": scanrefer_id,
                    "hash_id": hash_id,
                    "expected_response": expected_response,
                    "question_id": question_id,
                    "raw_question_id": raw_question_id,
                    "scan2cap_id": scan2cap_id,
                })
        
        logger.info(f"Generated {len(samples)} filter annotations across {len(self.scene_list)} scenes")
        return samples
    
    def _save_samples(self):
        pass

    def _load_samples(self):
        pass

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)


class Real3DFilterByAttributeDataset(Real3DDataset):
    """
    Dataset for filtering objects by attribute values.
    For each scene, generates one sample per unique attribute value.
    Each sample's result is all objects that have that specific attribute value.
    
    Example:
    - "Find all red objects" -> returns all objects with color=red
    - "Find all wooden objects" -> returns all objects with material=wood
    """
    
    def __init__(
        self,
        name: str = "scannet_attr_filter",
        attribute_file: str = f"{DATA_PATH}/mmscan-obj-desc/scannet_obj_infos/scannet_attribute.json",
        attributes_to_filter: List[str] = None,
        normalize_attributes: bool = True,
        max_objects_per_attribute: int = 50,  # Limit objects per attribute
        **kwargs
    ):
        """
        Initialize the attribute filter dataset.
        
        Args:
            name: Dataset name
            attribute_file: Path to attribute JSON file
            attributes_to_filter: List of attribute categories to use for filtering
            normalize_attributes: Whether to normalize multi-value attributes
            max_objects_per_attribute: Maximum objects to include per attribute value
            **kwargs: Other arguments for parent class
        """
        self.attribute_file = attribute_file
        self.normalize_attributes = normalize_attributes
        self.max_objects_per_attribute = max_objects_per_attribute
        
        # Default attribute categories to use for filtering
        self.attribute_categories = attributes_to_filter or [
            "color", "3D shape", "material", "usage", 
            "texture", "structure", "state"
        ]
        
        # Load attribute data
        self.attribute_data = self._load_attribute_data()
        
        # Override templates for this task
        kwargs["add_thinking_trace"] = kwargs.get("add_thinking_trace", False)
        kwargs["parallel"] = False
        
        super().__init__(name=name, **kwargs)
    
    def _load_attribute_data(self) -> Dict:
        """Load attribute data from JSON file."""
        logger.info(f"Loading attribute data from {self.attribute_file}")
        with open(self.attribute_file, 'r') as f:
            data = json.load(f)
        
        # Normalize attributes if needed
        if self.normalize_attributes:
            data = self._normalize_attribute_data(data)
        
        total_objects = sum(len(objects) for objects in data.values())
        logger.info(f"Loaded attributes for {total_objects} objects across {len(data)} scenes")
        
        return data
    
    def _normalize_attribute_data(self, data: Dict) -> Dict:
        """
        Normalize multi-value attributes to use consistent separator.
        Converts all separators (/, and, ,) to "or".
        """
        logger.info("Normalizing attribute data...")
        
        def parse_and_join(value: str) -> str:
            """Parse multi-value string and rejoin with 'or'."""
            if not value or not isinstance(value, str):
                return value
            
            # Split by various separators
            pattern = r'\s*[/,]\s*|\s+and\s+|\s+or\s+'
            values = re.split(pattern, value)
            
            # Clean and filter
            cleaned = [v.lower().strip() for v in values if v.strip()]
            
            # Rejoin with "or"
            return " or ".join(cleaned) if len(cleaned) > 1 else cleaned[0] if cleaned else ""
        
        normalized_data = {}
        for scene_id, objects in data.items():
            normalized_data[scene_id] = {}
            for obj_id, attributes in objects.items():
                normalized_data[scene_id][obj_id] = {}
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, str):
                        normalized_data[scene_id][obj_id][attr_name] = parse_and_join(attr_value)
                    elif isinstance(attr_value, list):
                        # Handle list values
                        normalized_values = [parse_and_join(v) if isinstance(v, str) else v for v in attr_value]
                        normalized_data[scene_id][obj_id][attr_name] = " or ".join(normalized_values)
                    else:
                        normalized_data[scene_id][obj_id][attr_name] = attr_value
        
        return normalized_data
    
    def _initialize_templates(self, **kwargs):
        """Initialize templates for attribute filtering task."""
        # Set instruction templates BEFORE calling super()
        kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
            "Identify all objects in the scene that have {attr_category} \"{attr_value}\". Provide their IDs, locations, and sizes.\n",
            "Find all objects with {attr_category} \"{attr_value}\". For each one, provide its ID, position, and dimensions.\n",
            "List all {attr_value} objects (by {attr_category}) with their IDs, coordinates, and sizes.\n",
            "Where are all the objects with {attr_category} \"{attr_value}\"? Give me their IDs, positions, and dimensions.\n",
            "Locate all objects that have {attr_category} \"{attr_value}\". For each one, specify its ID, location, and size.\n"
        ])
        
        # Call super to add common prefixes (thinking prompt + |object_set|)
        super()._initialize_templates(**kwargs)
        
        # Response templates
        self.response_templates = [
            "Apeiria found {count} object(s) with {attr_category} \"{attr_value}\" in the scene:\n{object_details}",
            "Roger. There are {count} object(s) with {attr_category} \"{attr_value}\" in this scene:\n{object_details}",
            "The scene contains {count} object(s) with {attr_category} \"{attr_value}\":\n{object_details}",
            "{count} object(s) with {attr_category} \"{attr_value}\" identified:\n{object_details}",
        ]
        
        # Thinking trace template
        self.thinking_trace_template = [
            "[APEIRIA THINKS]\n"
            "Apeiria will now analyze the scene to find objects with {attr_category} \"{attr_value}\".\n"
            "First, let me list all {object_count} objects and their details:\n"
            "{object_details_with_class}\n"
            "Now, I need to identify all objects with {attr_category} \"{attr_value}\". "
            "According to the analyzed object details, these objects match:\n"
            "{object_ids_with_class}\n"
            "Now, Apeiria will formulate the response based on the identified objects.\n"
            "[APEIRIA SPEAKS]\n"
        ]
        
        # Fix templates if needed
        if self.split != "train" or self.fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
            self.thinking_trace_template = [self.thinking_trace_template[0]]
    
    def _load_annotations(self):
        """Return empty list as we generate samples from scene data."""
        return []
    
    def _generate_samples(self, load_from_cache):
        """
        Generate samples: one per unique attribute value in each scene.
        Each sample filters objects by that attribute value.
        """
        logger.info(f"Generating attribute filter samples...")
        
        samples = []
        
        # Index to track attribute values across scenes
        # Structure: attr_category -> attr_value -> [(scene_id, obj_id, obj_idx), ...]
        attribute_index = defaultdict(lambda: defaultdict(list))
        
        # First pass: collect all attribute values and their associated objects
        for scene_id in self.scene_list:
            if scene_id not in self.scene_data:
                continue
            
            # Check if this scene has attribute data
            if scene_id not in self.attribute_data:
                continue
            
            # Get objects in this scene
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            object_ids = instance_bboxes[:, 7].astype(int)
            
            # For each object with attribute data
            for idx, obj_id in enumerate(object_ids):
                obj_id_str = str(obj_id)
                
                if obj_id_str not in self.attribute_data[scene_id]:
                    continue
                
                obj_attrs = self.attribute_data[scene_id][obj_id_str]
                
                # Index each attribute category and value
                for attr_category in self.attribute_categories:
                    if attr_category not in obj_attrs:
                        continue
                    
                    attr_value_str = obj_attrs[attr_category]
                    if not attr_value_str:
                        continue
                    
                    # Split by "or" to handle multi-value attributes
                    attr_values = [v.strip() for v in attr_value_str.split(" or ") if v.strip()]
                    
                    for attr_value in attr_values:
                        # Store the object info for this attribute value
                        attribute_index[attr_category][attr_value].append(
                            (scene_id, obj_id, idx)
                        )
        
        # Second pass: generate samples for each unique attribute value per scene
        for scene_id in self.scene_list:
            if scene_id not in self.scene_data:
                continue
            
            # Get scene info
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            locations = instance_bboxes[:, :6]
            object_classes = instance_bboxes[:, 6].astype(int)
            object_ids = instance_bboxes[:, 7].astype(int)
            num_objects = len(object_classes)
            
            # Track which attribute values we've already processed for this scene
            processed_in_scene = set()
            
            # Generate samples for each attribute category
            for attr_category in self.attribute_categories:
                # Get all attribute values for this category in this scene
                for attr_value, obj_list in attribute_index[attr_category].items():
                    # Filter to only objects in this scene
                    scene_objects = [
                        (oid, idx) for (sid, oid, idx) in obj_list 
                        if sid == scene_id
                    ]
                    
                    if not scene_objects:
                        continue
                    
                    # Check if we have too many objects (to avoid overly long responses)
                    if len(scene_objects) > self.max_objects_per_attribute:
                        continue
                    
                    # Create unique key to avoid duplicates
                    sample_key = f"{scene_id}|{attr_category}|{attr_value}"
                    if sample_key in processed_in_scene:
                        continue
                    processed_in_scene.add(sample_key)
                    
                    # Extract object IDs and indices
                    matched_object_ids = [oid for oid, _ in scene_objects]
                    matched_indices = [idx for _, idx in scene_objects]
                    
                    # Create instruction
                    description = random.choice(self.instruction_templates).format(
                        attr_category=attr_category,
                        attr_value=attr_value
                    )
                    
                    # Create object details
                    object_details = []
                    for idx in matched_indices:
                        obj_detail = random.choice(self.object_detail_templates).format(
                            id=object_ids[idx],
                            x=locations[idx, 0],
                            y=locations[idx, 1],
                            z=locations[idx, 2],
                            width=locations[idx, 3],
                            height=locations[idx, 4],
                            depth=locations[idx, 5]
                        )
                        object_details.append(obj_detail)
                    
                    object_details_str = "\n".join(object_details)
                    
                    # Create expected response
                    expected_response = random.choice(self.response_templates).format(
                        count=len(matched_indices),
                        attr_category=attr_category,
                        attr_value=attr_value,
                        object_details=object_details_str
                    )
                    
                    # Add thinking trace if enabled
                    if self.add_thinking_trace:
                        # Create detailed object descriptions with class names for all objects
                        object_details_with_class = []
                        for i in range(len(object_classes)):
                            obj_name = self.object_classes[int(object_classes[i])]
                            obj_id_with_class = self.object_id_with_class_templates.format(
                                id=object_ids[i],
                                object_name=obj_name
                            )
                            object_details_with_class.append(obj_id_with_class)
                        
                        object_details_with_class_str = ", ".join(object_details_with_class)
                        
                        # Create list of matched objects with class names
                        object_ids_with_class = []
                        for idx in matched_indices:
                            obj_name = self.object_classes[int(object_classes[idx])]
                            obj_id_with_class = self.object_id_with_class_templates.format(
                                id=object_ids[idx],
                                object_name=obj_name
                            )
                            object_ids_with_class.append(obj_id_with_class)
                        
                        # Format thinking trace
                        thinking_trace = random.choice(self.thinking_trace_template).format(
                            attr_category=attr_category,
                            attr_value=attr_value,
                            object_count=num_objects,
                            object_details_with_class=object_details_with_class_str,
                            object_ids_with_class=", ".join(object_ids_with_class)
                        )
                        
                        # Combine with response
                        expected_response = thinking_trace + expected_response
                    
                    # Create sample
                    ann_id = len(samples)
                    scanrefer_id = f"{scene_id}|{ann_id}"
                    hash_id = f"real_attr_filter_{scene_id}_{attr_category}_{attr_value}_{ann_id}"
                    scan2cap_id = f"{scene_id}|{attr_category}|{attr_value}"
                    
                    samples.append({
                        "scene_id": scene_id,
                        "ann_id": str(ann_id),
                        "description": description,
                        "program": f"filter_by_attribute(scene(), {attr_category}, {attr_value})",
                        "attr_category": attr_category,
                        "attr_value": attr_value,
                        "object_ids": matched_object_ids,
                        "object_id": matched_object_ids[0] if matched_object_ids else 0,
                        "scanrefer_id": scanrefer_id,
                        "hash_id": hash_id,
                        "expected_response": expected_response,
                        "question_id": scanrefer_id,
                        "raw_question_id": scanrefer_id,
                        "scan2cap_id": scan2cap_id,
                        "data_type": self.name,
                        "split": self.split,
                    })
        
        # Log statistics
        logger.info(f"Generated {len(samples)} attribute filter samples across {len(self.scene_list)} scenes")
        
        # Log per-category statistics
        category_counts = defaultdict(int)
        for sample in samples:
            category_counts[sample["attr_category"]] += 1
        
        logger.info("Samples per attribute category:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"  {category}: {count}")
        
        return samples

    def _save_samples(self):
        """Disable saving."""
        pass
    
    def _load_samples(self):
        """Disable loading from cache."""
        pass

    # Use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)


class Real3DDatasetFreeformThinking(Real3DDataset):
    """
    A dataset for freeform thinking traces from an external JSONL file.
    Each line in the file is a JSON object with "scene_id", "question", and "cot".
    The "cot" field contains <think> and <answer> tags.
    """
    MAX_COT_LENGTH = 1800 # about 99% percentile of the lengths; 99.5% ~ 1800
    def _get_annotation_file(self, split):
        data_dict = {
            "scene-r1": "data/Scene-30K.jsonl"
        }

        # NOTE: It includes train+val samples, but it is same with 3D-R1 code.

        return data_dict.get(split)

    def __init__(
        self,
        name: str = "scene-r1",
        use_trainval: bool = True, # this follows 3D-R1 code
        **kwargs
    ):
        """
        Initialize the dataset.

        Args:
            name (str): The name of the dataset.
            freeform_data_path (str): Path to the JSONL file with freeform thinking data.
            **kwargs: Other arguments for the parent Real3DDataset class.
        """
        self.freeform_data_path = self._get_annotation_file(name)
        if use_trainval and kwargs["split"] == "train":
            kwargs["split"] = "train,val"

        kwargs["add_thinking_trace"] = kwargs.get("add_thinking_trace", False) # shall be true
        # Disable parallel processing in parent during initial sample generation.
        kwargs["parallel"] = False
        kwargs["require_thinking_templates"] = ["Think about the scene freely before answering. "]

        logger.info(f"Initializing Real3DDatasetFreeformThinking with freeform_data_path: {self.freeform_data_path}")
        # Initialize parent class to load scene data, etc.
        super().__init__(name=name, **kwargs)

    def _load_annotations(self):
        # just reads the freeform data file
        annotations = []
        if not self.freeform_data_path:
            raise ValueError("`freeform_data_path` must be provided for Real3DDatasetFreeformThinking.")

        with open(self.freeform_data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                annotations.append(data)

        return annotations

    def _generate_samples(self, load_from_cache):
        """
        Generate samples by loading and processing the external freeform data file.
        """
        logger.info(f"Generating samples from freeform data file: {self.freeform_data_path}")
        samples = []

        with open(self.freeform_data_path, 'r') as f:
            # it includes both train and val samples. do filtering based on scene_id
            skipped_due_to_cot_length = 0
            for idx, line in enumerate(tqdm(f, desc="Processing freeform data")):
                try:
                    data = json.loads(line)
                    scene_id = data.get("scene_id")
                    question: str = data.get("question")
                    cot: str = data.get("cot")

                    if not all([scene_id, question, cot]):
                        logger.warning(f"Skipping line {idx+1} due to missing data.")
                        continue

                    if scene_id not in self.scene_data:
                        # logger.warning(f"Scene ID '{scene_id}' from line {idx+1} not found in loaded scene data. Skipping.")
                        # logger.warning(f"We have {len(self.scene_data)} scenes loaded: {self.scene_data.keys()}")
                        # raise
                        # might be of other splits
                        continue

                    # filter by max length
                    if self.MAX_COT_LENGTH > 0 and len(cot.split(' ')) > self.MAX_COT_LENGTH:
                        # logger.warning(f"Skipping line {idx+1} due to exceeding max COT length ({len(cot)} > {self.MAX_COT_LENGTH}).")
                        skipped_due_to_cot_length += 1
                        continue

                    # add "require thinking" and |object_set| to description
                    if self.add_thinking_trace:
                        question = random.choice(self.require_thinking_templates) + question

                    question = self.object_templates + question

                    # the question will be like:
                    # <Think about the scene first. > These are all objects in the scene: |object_set| <question> 
                    # This matches the format used of Real3DDataset: first the object set, then the instruction (question)

                    # Parse thinking trace and answer from the 'cot' field
                    think_match = re.search(r"<think>(.*?)</think>", cot, re.DOTALL)
                    answer_match = re.search(r"<answer>(.*?)</answer>", cot, re.DOTALL)

                    if not think_match or not answer_match:
                        logger.warning(f"Could not parse <think> or <answer> tags in line {idx+1}. Skipping.")
                        continue
                    
                    thinking_content = think_match.group(1).strip()
                    answer_content = answer_match.group(1).strip()

                    # Format the thinking trace and final response
                    thinking_trace = f"[APEIRIA THINKS]\n{thinking_content}\n[APEIRIA SPEAKS]"
                    expected_response = f"{thinking_trace}\n{answer_content}"

                    # Create a unique ID for the sample
                    scanrefer_id = f"{scene_id}|freeform_{idx}"
                    
                    # For this dataset, we don't have a specific ground truth object_id.
                    # We can set it to a placeholder value.
                    object_id = 0 # for compatibility

                    samples.append({
                        "scene_id": scene_id,
                        "description": question, # The 'question' is the prompt
                        "raw_description": question,
                        "program": "", # No program for freeform thinking
                        "object_id": object_id,
                        "object_ids": [object_id],
                        "result_objects": [], # No specific result objects
                        "objects": [],
                        "scanrefer_id": scanrefer_id,
                        "hash_id": f"real_freeform_{scanrefer_id}",
                        "question_id": scanrefer_id,
                        "raw_question_id": scanrefer_id,
                        "scan2cap_id": f"{scene_id}|{object_id}|freeform",
                        "data_type": self.name,
                        "split": self.split,
                        "expected_response": expected_response,
                        "thinking_trace_parts": { # Store parts for potential use
                            "all": expected_response,
                            "plan": None, # No structured plan
                            "execution": thinking_content,
                            "header": f"[APEIRIA THINKS]\nI need to answer the question: \"{question}\""
                        },
                        "prompt_with_plan": None, # No plan to include in prompt
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {idx+1}.")
                except Exception as e:
                    logger.error(f"Error processing line {idx+1}: {e}")
                    print_exc()

        logger.info(f"Generated {len(samples)} samples from the freeform data file {self.freeform_data_path}.")
        logger.info(f"Skipped {skipped_due_to_cot_length} samples due to exceeding max COT length of {self.MAX_COT_LENGTH}.")
        return samples

    
    def evaluate(self, preds: Dict[str, str], gt_indices: Dict, hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Evaluation for freeform thinking using text-based metrics (BLEU, ROUGE, CIDEr, METEOR).
        
        Args:
            preds: Dictionary mapping sample ID to predicted response text.
            gt_indices: Not used for this evaluation, but kept for compatibility.
            hash_id_index: Whether keys in preds are hash_ids instead of scanrefer_ids.
            **kwargs: Other unused arguments.
            
        Returns:
            message: Evaluation summary message.
            metrics: Dictionary of evaluation metrics.
        """
        logger.info("Evaluating freeform thinking responses with text metrics...")
        
        candidates = {}
        corpus = {}

        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # This dataset does not use hash_id_index logic from parent,
                # but we can build a temporary map if needed.
                # For now, assume key is scanrefer_id.
                pass

            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample with ID '{scanrefer_id}' not found in dataset. Skipping.")
                continue

            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            # Extract the answer part from the prediction.
            # The parse_response function in the parent module can do this.
            parsed_pred_objects = parse_response(pred_text) # This also returns the text after [APEIRIA SPEAKS]
            if re.search(r"\[APEIRIA SPEAKS\]", pred_text, flags=re.IGNORECASE):
                pred_answer = re.split(r"\[APEIRIA SPEAKS\]", pred_text, flags=re.IGNORECASE)[-1].strip()
            else:
                pred_answer = pred_text.strip()

            # Extract the ground truth answer from the sample's expected_response.
            gt_response = sample["expected_response"]
            if re.search(r"\[APEIRIA SPEAKS\]", gt_response, flags=re.IGNORECASE):
                gt_answer = re.split(r"\[APEIRIA SPEAKS\]", gt_response, flags=re.IGNORECASE)[-1].strip()
            else:
                # Fallback if the GT format is unexpected
                gt_answer = gt_response.strip()

            candidates[scanrefer_id] = [pred_answer]
            corpus[scanrefer_id] = [gt_answer]

        if not candidates:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}

        # Use the provided eval_utils to calculate scores
        score_per_caption, message, metrics = score_captions(corpus, candidates)
        
        # Add total samples evaluated to metrics
        metrics["total_evaluated"] = len(candidates)
        
        full_message = f"Freeform Response Evaluation ({len(candidates)} samples):\n{message}"
        
        return full_message, metrics
        
    
    def _save_samples(self):
        # Saving is not recommended due to large size, but can be implemented if needed.
        pass

    def _load_samples(self):
        # Loading from cache is disabled for this class.
        pass

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)


class Real3DDatasetWithExternalTrace(Real3DDataset):
    """
    A child class of Real3DDataset that incorporates external thinking traces
    instead of generating them internally. Supports multiple traces per sample.
    """
    
    def __init__(
        self,
        external_traces_path: str = None,
        external_traces_dict: Dict[str, List[Dict]] = None,
        shuffle_traces: bool = True,
        max_traces_per_sample: int = None,
        **kwargs
    ):
        """
        Initialize the dataset with external thinking traces.
        
        Args:
            external_traces_path: Path to JSON file containing external thinking traces
            external_traces_dict: Dictionary mapping sample keys to lists of thinking traces
            shuffle_traces: Whether to shuffle traces for each sample during loading
            max_traces_per_sample: Maximum number of traces to use per sample (None = use all)
            **kwargs: Other arguments passed to parent class

            Expected external JSON:
            ```json
            {
                "scene0000_00|0": [
                    {
                    "thinking_trace": "第一种思考方式...",
                    "thinking_trace_parts": {...}
                    },
                    {
                    "thinking_trace": "第二种思考方式...",
                    "thinking_trace_parts": {...}
                    }
                ],
                "scene0000_00|1": [],  // 空列表，这个sample会被移除
                "scene0000_00|2": [    // 只有一个trace
                    {
                    "thinking_trace": "...",
                    "thinking_trace_parts": {...}
                    }
                ]
            }
            ```
        """
        self.external_traces_path = external_traces_path
        self.external_traces_dict = external_traces_dict
        self.shuffle_traces = shuffle_traces
        self.max_traces_per_sample = max_traces_per_sample
        kwargs["parallel"] = False  # Disable parallel in parent during initial generation
        
        # Load external traces if path is provided
        if self.external_traces_path and not self.external_traces_dict:
            self.external_traces_dict = self._load_external_traces(self.external_traces_path)
        
        # Initialize parent class
        super().__init__(**kwargs)
    
    def _load_external_traces(self, path: str) -> Dict[str, List[Dict]]:
        """Load external thinking traces from JSON file."""
        logger.info(f"Loading external thinking traces from {path}")
        with open(path, 'r') as f:
            traces = json.load(f)
        
        # Count total traces
        total_traces = sum(len(trace_list) for trace_list in traces.values())
        logger.info(f"Loaded {total_traces} external thinking traces for {len(traces)} unique samples")
        
        return traces
    
    def _generate_samples_parallel(self, load_from_cache):
        """Override to use external thinking traces instead of generating them."""
        if load_from_cache:
            return self._load_samples()

        # First, generate samples using parent method but with thinking trace disabled
        original_add_thinking_trace = self.add_thinking_trace
        self.add_thinking_trace = False  # Temporarily disable to avoid generation
        
        samples = super()._generate_samples_parallel(load_from_cache=False)
        
        self.add_thinking_trace = original_add_thinking_trace  # Restore
        
        # Now incorporate external thinking traces
        if self.external_traces_dict and self.add_thinking_trace:
            samples = self._incorporate_external_traces(samples)
        
        return samples
    
    def _incorporate_external_traces(self, samples: List[Dict]) -> List[Dict]:
        """
        Incorporate external thinking traces into samples.
        Expands samples based on number of available traces.
        """
        logger.info("Incorporating external thinking traces into samples...")
        
        expanded_samples = []
        removed_count = 0
        
        for sample in samples:
            # Create a unique key to match with external traces
            trace_key = self._create_trace_key(sample)
            
            if trace_key in self.external_traces_dict:
                trace_list = self.external_traces_dict[trace_key]
                
                # Skip if no traces available
                if not trace_list:
                    removed_count += 1
                    continue
                
                # Optionally shuffle traces
                if self.shuffle_traces:
                    trace_list = random.sample(trace_list, len(trace_list))
                
                # Limit number of traces if specified
                if self.max_traces_per_sample is not None:
                    trace_list = trace_list[:self.max_traces_per_sample]
                
                # Create one sample for each trace
                for trace_idx, external_trace in enumerate(trace_list):
                    # Deep copy the sample to avoid modifying the original
                    new_sample = copy.deepcopy(sample)
                    
                    # Extract thinking trace and parts
                    thinking_trace = external_trace.get("thinking_trace", "")
                    thinking_parts = external_trace.get("thinking_trace_parts", {})
                    
                    # Get the response body (everything after thinking trace in expected_response)
                    response_body = new_sample["expected_response"]
                    
                    # Combine thinking trace with response body
                    if thinking_trace:
                        new_sample["expected_response"] = thinking_trace + "\n" + response_body
                        new_sample["thinking_trace_parts"] = thinking_parts
                    
                    # Add prompt_with_plan if plan is available
                    if "plan" in thinking_parts and self.tokenizer:
                        plan_with_start = thinking_parts["header"] + "\n" + thinking_parts["plan"]
                        new_sample["prompt_with_plan"] = apply_qwen_template_with_partial_response(
                            new_sample["description"], 
                            self.tokenizer, 
                            plan_with_start,
                        )[0]
                    
                    # Add trace metadata
                    new_sample["trace_idx"] = trace_idx
                    new_sample["total_traces"] = len(trace_list)
                    new_sample["original_scanrefer_id"] = new_sample["scanrefer_id"]
                    
                    # Update IDs to make them unique
                    new_sample["scanrefer_id"] = f"{new_sample['scanrefer_id']}_trace{trace_idx}"
                    new_sample["hash_id"] = f"{new_sample['hash_id']}_trace{trace_idx}"
                    new_sample["question_id"] = f"{new_sample['question_id']}_trace{trace_idx}"
                    new_sample["raw_question_id"] = f"{new_sample['raw_question_id']}_trace{trace_idx}"
                    
                    expanded_samples.append(new_sample)
            else:
                # No external traces found - remove this sample
                removed_count += 1
                logger.debug(f"No external thinking trace found for key: {trace_key}")
        
        logger.info(f"Expanded {len(samples)} samples to {len(expanded_samples)} samples")
        logger.info(f"Removed {removed_count} samples without external traces")
        
        return expanded_samples
    
    def _create_trace_key(self, sample: Dict) -> str:
        """
        Create a unique key for matching samples with external traces.
        Can be overridden for different key formats.
        """
        # Use scanrefer_id as the default key
        return sample["scanrefer_id"]
    
    def save_external_trace_template(self, output_path: str):
        """
        Save a template file showing the expected format for external thinking traces.
        Now supports multiple traces per sample.
        """
        template = {
            "format_description": "External thinking traces for Real3DDataset (multi-trace version)",
            "version": "2.0",
            "example_entries": {
                "scene0000_00|0": [
                    {
                        "thinking_trace": "[APEIRIA THINKS]\nI need to find the object described as: \"the chair next to the table\"\nFirst approach: Let me start by finding all chairs...\n[APEIRIA SPEAKS]",
                        "thinking_trace_parts": {
                            "all": ["[APEIRIA THINKS]", "..."],
                            "plan": "Let's plan my next steps: Find all chairs first; Then check spatial relations",
                            "execution": "First approach: Let me start by finding all chairs...",
                            "header": "[APEIRIA THINKS]\nI need to find the object described as: \"the chair next to the table\""
                        }
                    },
                    {
                        "thinking_trace": "[APEIRIA THINKS]\nI need to find the object described as: \"the chair next to the table\"\nAlternative approach: Let me first identify all tables...\n[APEIRIA SPEAKS]",
                        "thinking_trace_parts": {
                            "all": ["[APEIRIA THINKS]", "..."],
                            "plan": "Let's plan my next steps: Find all tables; Then find chairs near them",
                            "execution": "Alternative approach: Let me first identify all tables...",
                            "header": "[APEIRIA THINKS]\nI need to find the object described as: \"the chair next to the table\""
                        }
                    }
                ],
                "scene0000_00|1": [],  # Empty list means this sample will be removed
                "scene0000_00|2": [
                    {
                        "thinking_trace": "[APEIRIA THINKS]\n...\n[APEIRIA SPEAKS]",
                        "thinking_trace_parts": {
                            "all": ["..."],
                            "plan": "...",
                            "execution": "...",
                            "header": "..."
                        }
                    }
                ]
            },
            "required_fields": {
                "per_trace": {
                    "thinking_trace": "The complete thinking trace text",
                    "thinking_trace_parts": "Dictionary containing trace parts"
                }
            },
            "notes": [
                "Each key maps to a LIST of thinking traces",
                "Empty list means the sample will be removed from dataset",
                "Each trace in the list will create a separate training sample",
                "Keys should match the scanrefer_id format: {scene_id}|{annotation_index}"
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Saved external trace template to {output_path}")

# [deprecated]
class Real3DDatasetWithAttributes(Real3DDataset):
    """
    Dataset for object attribute description tasks.
    Supports two modes:
    1. Dense captioning: Generate descriptions for all objects in a scene, using ScanRefer/NR3D/SR3D data
    2. Attribute captioning: Describe specific attributes for all objects
    """
    
    def __init__(
        self,
        name: str = "scannet_attributes", # or "sr3d_dense", "nr3d_dense" or "scanrefer_dense"
        attribute_file: str = f"{DATA_PATH}/mmscan-obj-desc/scannet_obj_infos/scannet_attribute.json",
        attributes_to_describe: List[str] = None,
        normalize_attributes: bool = True,
        max_captions_per_object: int = 10,  # Limit number of captions per object
        **kwargs
    ):
        """
        Initialize the attribute description dataset.
        
        Args:
            name: Dataset name (use "dense" suffix for dense captioning mode)
            attribute_file: Path to attribute JSON file
            attributes_to_describe: List of attribute categories to describe
            normalize_attributes: Whether to normalize multi-value attributes
            max_captions_per_object: Maximum number of captions to use per object (None = use all)
            **kwargs: Other arguments for parent class
        """
        self.attribute_file = attribute_file
        self.mode = "dense" if "dense" in name else "attributes"
        self.normalize_attributes = normalize_attributes
        self.max_captions_per_object = max_captions_per_object
        
        # Default attribute categories
        self.attribute_categories = attributes_to_describe or [
            "color", "3D shape", "material", "usage", 
            "texture", "structure", "state"
        ]
        
        # For attributes mode, load attribute data
        # For dense mode, we'll collect captions from annotations in _load_annotations
        if self.mode == "attributes":
            self.attribute_data = self._load_attribute_data()
        
        # Override templates for this task
        kwargs["add_thinking_trace"] = False
        kwargs["parallel"] = False
        
        super().__init__(name=name, **kwargs)

    def _load_attribute_data(self) -> Dict:
        """Load attribute data from JSON file."""
        logger.info(f"Loading attribute data from {self.attribute_file}")
        with open(self.attribute_file, 'r') as f:
            data = json.load(f)
        
        # Normalize attributes if needed
        if self.normalize_attributes:
            data = self._normalize_attribute_data(data)
        
        total_objects = sum(len(objects) for objects in data.values())
        logger.info(f"Loaded attributes for {total_objects} objects across {len(data)} scenes")
        
        return data
    
    def _normalize_attribute_data(self, data: Dict) -> Dict:
        """
        Normalize multi-value attributes to use consistent separator.
        Converts all separators (/, and, ,) to "or".
        """
        logger.info("Normalizing attribute data...")
        
        def parse_and_join(value: str) -> str:
            """Parse multi-value string and rejoin with 'or'."""
            if not value or not isinstance(value, str):
                return value
            
            # Split by various separators
            pattern = r'\s*[/,]\s*|\s+and\s+|\s+or\s+'
            values = re.split(pattern, value)
            
            # Clean and filter
            cleaned = [v.lower().strip() for v in values if v.strip()]
            
            # Rejoin with "or"
            return " or ".join(cleaned) if len(cleaned) > 1 else cleaned[0] if cleaned else ""
        
        normalized_data = {}
        for scene_id, objects in data.items():
            normalized_data[scene_id] = {}
            for obj_id, attributes in objects.items():
                normalized_data[scene_id][obj_id] = {}
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, str):
                        normalized_data[scene_id][obj_id][attr_name] = parse_and_join(attr_value)
                    elif isinstance(attr_value, list):
                        # Handle list values
                        normalized_values = [parse_and_join(v) if isinstance(v, str) else v for v in attr_value]
                        normalized_data[scene_id][obj_id][attr_name] = " or ".join(normalized_values)
                    else:
                        normalized_data[scene_id][obj_id][attr_name] = attr_value
        
        return normalized_data
    
    def _initialize_templates(self, **kwargs):
        """Initialize templates for attribute/dense description task."""
        super()._initialize_templates(**kwargs)
        
        if self.mode == "attributes":
            # Attribute captioning templates
            self.instruction_templates = [
                "Describe the attributes of all objects in the scene. For each object, provide its ID and the following attributes: {attributes}.\n",
                "For each object in this scene, specify its ID and describe these attributes: {attributes}.\n",
                "List all objects with their IDs and describe their {attributes}.\n",
                "Provide attribute descriptions for all objects. Include ID and {attributes} for each.\n",
            ]
            
            # Template for single object attribute description
            self.object_attr_templates = [
                "Object {id}({object_name}): {attributes}",
                "ID {id}({object_name}) - {attributes}",
                "{id}: {object_name}. {attributes}",
            ]
            
            # Template for single attribute
            self.single_attr_templates = [
                "{attr_name}: {attr_value}",
                "{attr_name} is {attr_value}",
            ]

            # choose one if not train
            if self.split != "train" or self.fix_template:
                self.instruction_templates = [self.instruction_templates[0]]
                self.object_attr_templates = [self.object_attr_templates[0]]
                self.single_attr_templates = [self.single_attr_templates[0]]
            
        else:  # dense captioning mode
            self.instruction_templates = [
                "Describe all objects in the scene. For each object, provide its ID, type and description(s).\n",
                "Generate descriptions for all objects. Include object ID, name and captions.\n",
                "List all objects with their IDs, names, and detailed descriptions.\n",
                "Provide comprehensive descriptions for each object in the scene. Include ID, object type, and all known descriptions.\n",
            ]
            
            # Template for objects with multiple captions
            self.object_desc_templates = [
                "Object {id}({object_name}): {description}",
                "ID {id}({object_name}) - {description}",
                "{id}: {object_name}. {description}",
                # "Object {id} is a {object_name}. {description}",
            ]
            
            # Template for joining multiple captions
            self.caption_joiner_templates = [
                "; ",  # Semicolon separator
                " | ",  # Pipe separator
                "\n  - ",  # Bulleted list style
            ]

            if self.split != "train" or self.fix_template:
                self.instruction_templates = [self.instruction_templates[0]]
                self.object_desc_templates = [self.object_desc_templates[0]]
                self.caption_joiner_templates = [self.caption_joiner_templates[0]]
        
        # Response templates (common for both modes)
        self.response_templates = [
            "Apeiria found {count} objects in the scene:\n{object_details}",
            "Roger. There are {count} objects in this scene:\n{object_details}",
            "The scene contains {count} objects:\n{object_details}",
            "Apeiria has identified {count} objects:\n{object_details}",
        ]

        if self.split != "train" or self.fix_template:
            self.response_templates = [self.response_templates[0]]

    
    def _load_annotations(self):
        """
        Override to collect dense captions for dense mode.
        For attributes mode, just return empty list.
        """
        if self.mode == "dense":
            # Load original annotations from SR3D/NR3D for dense captioning
            annotation_file = self._get_annotation_file(self.split)
            logger.info(f"Loading dense caption annotations from {annotation_file}")
            
            with open(annotation_file, 'r') as f:
                raw_annotations = json.load(f)
            
            # Collect all descriptions for each object
            self.dense_captions = self._collect_dense_captions(raw_annotations)
            
            total_objects = sum(len(objs) for objs in self.dense_captions.values())
            total_captions = sum(
                len(captions) 
                for scene_objs in self.dense_captions.values() 
                for captions in scene_objs.values()
            )
            logger.info(f"[{self.name} ({self.__class__.__name__})] Collected {total_captions} captions for {total_objects} unique objects across {len(self.dense_captions)} scenes")
        
        # Return empty list as we'll generate samples in _generate_samples
        return []

    def _collect_dense_captions(self, annotations: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
        """
        Collect all captions for each object from annotations.
        
        Args:
            annotations: List of annotation dictionaries from SR3D/NR3D
            
        Returns:
            Dictionary mapping scene_id -> object_id (str) -> list of captions
        """
        captions = defaultdict(lambda: defaultdict(list))
        
        for anno in annotations:
            scene_id = anno["scene_id"]
            object_id = str(anno["object_id"])
            description = anno["description"].strip()
            
            if scene_id and object_id and description:
                captions[scene_id][object_id].append(description)
        
        # Convert to regular dict and optionally limit captions per object
        result = {}
        for scene_id, scene_objs in captions.items():
            result[scene_id] = {}
            for obj_id, caption_list in scene_objs.items():
                if self.max_captions_per_object is not None:
                    # Take random sample if limiting
                    if len(caption_list) > self.max_captions_per_object:
                        caption_list = random.sample(caption_list, self.max_captions_per_object)
                result[scene_id][obj_id] = caption_list
        
        return result
    
    def _generate_samples(self, load_from_cache):
        """Generate samples for attribute/dense description task."""
        logger.info(f"Generating {self.mode} captioning samples...")
        
        samples = []
        
        for scene_id in self.scene_list:
            if scene_id not in self.scene_data:
                continue
            
            # Get objects in this scene
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            object_ids = instance_bboxes[:, 7].astype(int)
            
            if self.mode == "dense":
                # Dense captioning mode
                if scene_id not in self.dense_captions:
                    logger.debug(f"No dense captions found for scene {scene_id}")
                    continue
                
                # Create instruction
                description = random.choice(self.instruction_templates)
                
                # Generate object descriptions
                object_details = []
                valid_object_count = 0
                
                for idx, obj_id in enumerate(object_ids):
                    obj_id_str = str(obj_id)
                    object_name = self.object_classes[int(instance_bboxes[idx, 6])]
                    
                    # Check if this object has captions
                    if obj_id_str not in self.dense_captions[scene_id]:
                        continue
                    
                    # Get all captions for this object
                    obj_captions = self.dense_captions[scene_id][obj_id_str]
                    
                    if not obj_captions:
                        continue
                    
                    # Format captions
                    if len(obj_captions) == 1:
                        # Single caption
                        caption_text = obj_captions[0]
                    else:
                        # Multiple captions - join them
                        joiner = random.choice(self.caption_joiner_templates)
                        if "\n" in joiner:  # Bulleted list style
                            caption_text = joiner + joiner.join(obj_captions)
                        else:
                            caption_text = joiner.join(obj_captions)
                    
                    # Create object detail
                    obj_detail = random.choice(self.object_desc_templates).format(
                        id=obj_id,
                        object_name=object_name,
                        description=caption_text
                    )
                    
                    object_details.append(obj_detail)
                    valid_object_count += 1
                
                if not object_details:
                    continue
                
                # Create expected response
                object_details_str = "\n".join(object_details)
                expected_response = random.choice(self.response_templates).format(
                    count=valid_object_count,
                    object_details=object_details_str
                )
                
            else:  # attributes mode
                # Filter objects that have attribute data
                if scene_id not in self.attribute_data:
                    continue
                
                # Create instruction
                attr_list = ", ".join(self.attribute_categories)
                description = random.choice(self.instruction_templates).format(attributes=attr_list)
                
                # Generate object descriptions
                object_details = []
                valid_object_count = 0
                
                for idx, obj_id in enumerate(object_ids):
                    obj_id_str = str(obj_id)
                    object_name = self.object_classes[int(instance_bboxes[idx, 6])]
                    
                    # Check if this object has attribute data
                    if obj_id_str not in self.attribute_data[scene_id]:
                        continue
                    
                    # Get attributes for this object
                    obj_attrs = self.attribute_data[scene_id][obj_id_str]
                    
                    # Format attributes
                    attr_strings = []
                    for attr_name in self.attribute_categories:
                        if attr_name in obj_attrs:
                            attr_value = obj_attrs[attr_name]
                            attr_str = random.choice(self.single_attr_templates).format(
                                attr_name=attr_name,
                                attr_value=attr_value
                            )
                            attr_strings.append(attr_str)
                    
                    if not attr_strings:
                        continue
                    
                    # Combine attributes
                    combined_attrs = ", ".join(attr_strings)
                    obj_detail = random.choice(self.object_attr_templates).format(
                        id=obj_id,
                        object_name=object_name,
                        attributes=combined_attrs
                    )
                    
                    object_details.append(obj_detail)
                    valid_object_count += 1
                
                if not object_details:
                    continue
                
                # Create expected response
                object_details_str = "\n".join(object_details)
                expected_response = random.choice(self.response_templates).format(
                    count=valid_object_count,
                    object_details=object_details_str
                )
            
            # Create sample (common for both modes)
            ann_id = len(samples)
            scanrefer_id = f"{scene_id}|{ann_id}"
            hash_id = f"real_{self.mode}_{scene_id}_{ann_id}"
            
            samples.append({
                "scene_id": scene_id,
                "ann_id": str(ann_id),
                "description": description,
                "program": "",  # No program for this task
                "object_ids": object_ids.tolist(),
                "object_id": object_ids[0] if len(object_ids) > 0 else 0,
                "scanrefer_id": scanrefer_id,
                "hash_id": hash_id,
                "expected_response": expected_response,
                "question_id": scanrefer_id,
                "raw_question_id": scanrefer_id,
                "scan2cap_id": f"{scene_id}|{self.mode}",
                "data_type": self.name,
                "split": self.split,
            })
        
        logger.info(f"Generated {len(samples)} {self.mode} captioning samples")
        return samples
    
    def evaluate(self, preds: Dict[str, str], gt_indices: Dict, 
             hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Evaluate attribute descriptions using text-based metrics.
        """
        logger.info("Evaluating attribute descriptions with text metrics...")
        
        candidates = {}
        corpus = {}
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id if needed
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample with ID '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            candidates[scanrefer_id] = [pred_text]
            corpus[scanrefer_id] = [sample["expected_response"]]
        
        if not candidates:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}

        if self.mode == "dense":
            logger.info("Using dense captioning evaluation metrics.")
        
            # Use eval_utils to calculate scores
            score_per_caption, message, metrics = score_captions(corpus, candidates)
            metrics["total_evaluated"] = len(candidates)

        else:  # attributes mode
            # Test if the predicted attributes match the ground truth attributes
            logger.info("Using attribute description evaluation metrics.")
            
            # Metrics per attribute category
            attr_metrics = {attr_name: {
                'tp': 0,  # true positives
                'fp': 0,  # false positives
                'fn': 0,  # false negatives
                'total_pred': 0,  # total predicted values
                'total_gt': 0,    # total ground truth values
            } for attr_name in self.attribute_categories}
            
            # Overall metrics
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for scanrefer_id in candidates:
                pred_text = candidates[scanrefer_id][0].lower()
                gt_text = corpus[scanrefer_id][0].lower()
                
                # Get the sample to access scene_id and object_ids
                sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
                scene_id = sample["scene_id"]
                
                # Parse predicted attributes from text
                # Expected format: "Object {id}({name}): attr1: value1, attr2: value2, ..."
                pred_attrs_by_object = self._parse_attribute_response(pred_text)
                # => dict of object_id -> attr_name -> list of values
                
                # Get ground truth attributes for this scene
                gt_attrs_by_object = {}
                if scene_id in self.attribute_data:
                    for obj_id_str, obj_attrs in self.attribute_data[scene_id].items():
                        gt_attrs_by_object[obj_id_str] = obj_attrs
                
                # Compare attributes for each object
                for obj_id_str, pred_obj_attrs in pred_attrs_by_object.items():
                    if obj_id_str not in gt_attrs_by_object:
                        # Predicted object not in ground truth
                        for attr_name, pred_values in pred_obj_attrs.items():
                            if attr_name in attr_metrics:
                                attr_metrics[attr_name]['fp'] += len(pred_values)
                                attr_metrics[attr_name]['total_pred'] += len(pred_values)
                                total_fp += len(pred_values)
                        continue
                    
                    gt_obj_attrs = gt_attrs_by_object[obj_id_str]
                    
                    # Check each attribute category
                    for attr_name in self.attribute_categories:
                        pred_values = set(pred_obj_attrs.get(attr_name, []))
                        
                        # Parse ground truth values (split by "or")
                        gt_value_str = gt_obj_attrs.get(attr_name, "")
                        gt_values = set()
                        if gt_value_str:
                            # Split by "or" and clean
                            gt_values = set(v.strip().lower() for v in gt_value_str.split(" or ") if v.strip())
                        
                        # Update counts
                        attr_metrics[attr_name]['total_pred'] += len(pred_values)
                        attr_metrics[attr_name]['total_gt'] += len(gt_values)
                        
                        # Calculate matches
                        matches = pred_values & gt_values  # intersection
                        tp = len(matches)
                        fp = len(pred_values - gt_values)
                        fn = len(gt_values - pred_values)
                        
                        attr_metrics[attr_name]['tp'] += tp
                        attr_metrics[attr_name]['fp'] += fp
                        attr_metrics[attr_name]['fn'] += fn
                        
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                
                # Check for objects in GT but not in prediction
                for obj_id_str, gt_obj_attrs in gt_attrs_by_object.items():
                    if obj_id_str not in pred_attrs_by_object:
                        # Ground truth object not predicted
                        for attr_name in self.attribute_categories:
                            gt_value_str = gt_obj_attrs.get(attr_name, "")
                            if gt_value_str:
                                gt_values = set(v.strip().lower() for v in gt_value_str.split(" or ") if v.strip())
                                attr_metrics[attr_name]['fn'] += len(gt_values)
                                attr_metrics[attr_name]['total_gt'] += len(gt_values)
                                total_fn += len(gt_values)
            
            # Calculate metrics
            def calc_metrics(tp, fp, fn):
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                return precision, recall, f1
            
            # Overall metrics
            overall_precision, overall_recall, overall_f1 = calc_metrics(total_tp, total_fp, total_fn)
            
            # Per-attribute metrics
            attr_results = {}
            for attr_name, counts in attr_metrics.items():
                precision, recall, f1 = calc_metrics(counts['tp'], counts['fp'], counts['fn'])
                attr_results[attr_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': counts['tp'],
                    'fp': counts['fp'],
                    'fn': counts['fn'],
                    'total_pred': counts['total_pred'],
                    'total_gt': counts['total_gt'],
                }
            
            # Build metrics dictionary
            metrics = {
                'overall_precision': overall_precision,
                'overall_recall': overall_recall,
                'overall_f1': overall_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn,
                'total_evaluated': len(candidates),
            }
            
            # Add per-attribute metrics
            for attr_name, attr_metric in attr_results.items():
                metrics[f'{attr_name}_precision'] = attr_metric['precision']
                metrics[f'{attr_name}_recall'] = attr_metric['recall']
                metrics[f'{attr_name}_f1'] = attr_metric['f1']
            
            # Build message
            message_lines = [
                f"Overall Metrics:",
                f"  Precision: {overall_precision:.4f}",
                f"  Recall: {overall_recall:.4f}",
                f"  F1 Score: {overall_f1:.4f}",
                f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}",
                f"\nPer-Attribute Metrics:"
            ]
            
            for attr_name in self.attribute_categories:
                attr_metric = attr_results[attr_name]
                message_lines.append(
                    f"  {attr_name:15s}: "
                    f"P={attr_metric['precision']:.3f}, "
                    f"R={attr_metric['recall']:.3f}, "
                    f"F1={attr_metric['f1']:.3f} "
                    f"(TP={attr_metric['tp']}, FP={attr_metric['fp']}, FN={attr_metric['fn']})"
                )
            
            message = "\n".join(message_lines)
        
        full_message = f"Attribute Description Evaluation ({len(candidates)} samples):\n{message}"
        
        return full_message, metrics

    def _parse_attribute_response(self, response_text: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Parse attribute descriptions from model response.
        
        Args:
            response_text: Model's response text
            
        Returns:
            Dictionary mapping object_id -> attribute_name -> list of values
            
        Example:
            Input: "Object 1(chair): color: black or white, material: wood"
            Output: {"1": {"color": ["black", "white"], "material": ["wood"]}}
        """
        attrs_by_object = {}
        
        # Split by lines to get individual object descriptions
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match pattern: "Object {id}({name}): ..." or "ID {id}({name}) - ..." or "{id}: {name}. ..."
            # Pattern 1: Object 1(chair): ...
            match = re.match(r'(?:object\s+)?(\d+)\s*\([^)]+\)\s*[:\-]\s*(.+)', line, re.IGNORECASE)
            if not match:
                # Pattern 2: 1: chair. ...
                match = re.match(r'(\d+)\s*:\s*[^.]+\.\s*(.+)', line, re.IGNORECASE)
            
            if not match:
                continue
            
            obj_id = match.group(1)
            attr_text = match.group(2)
            
            # Parse attributes from the text
            # Expected format: "attr1: value1, attr2: value2 or value3, ..."
            obj_attrs = {}
            
            # Split by commas (but be careful with "or" inside values)
            # Use a more sophisticated approach
            attr_parts = []
            current_part = ""
            paren_depth = 0
            
            for char in attr_text:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == ',' and paren_depth == 0:
                    attr_parts.append(current_part.strip())
                    current_part = ""
                    continue
                current_part += char
            
            if current_part.strip():
                attr_parts.append(current_part.strip())
            
            # Parse each attribute part
            for part in attr_parts:
                # Match "attr_name: value" or "attr_name is value"
                attr_match = re.match(r'([^:]+?)(?:\s+is\s+|\s*:\s*)(.+)', part, re.IGNORECASE)
                if not attr_match:
                    continue
                
                attr_name = attr_match.group(1).strip().lower()
                attr_value = attr_match.group(2).strip().lower()
                
                # Check if this is one of our attribute categories
                # (allow partial matching)
                matched_category = None
                for category in self.attribute_categories:
                    if category.lower() in attr_name or attr_name in category.lower():
                        matched_category = category
                        break
                
                if not matched_category:
                    continue
                
                # Split values by "or"
                values = [v.strip() for v in attr_value.split(" or ") if v.strip()]
                
                obj_attrs[matched_category] = values
            
            if obj_attrs:
                attrs_by_object[obj_id] = obj_attrs
        
        return attrs_by_object

    
    def _save_samples(self):
        """Disable saving for this dataset."""
        pass
    
    def _load_samples(self):
        """Disable loading from cache."""
        pass

class Real3DDatasetWithAttributesNew(Real3DDataset):
    """
    Dataset for object attribute description tasks.
    Now generates one sample per object, similar to Real3DObjectInfoDataset.
    Each sample describes attributes for a single object.

    NOTE: not all objects have attribute data, so it might have fewer samples than objects.
    """
    
    def __init__(
        self,
        name: str = "scannet_attributes",
        attribute_file: str = f"{DATA_PATH}/mmscan-obj-desc/scannet_obj_infos/scannet_attribute.json",
        attributes_to_describe: List[str] = None,
        normalize_attributes: bool = True,
        no_object_id_input: bool = False,  # Whether to hide object ID in the prompt
        **kwargs
    ):
        """
        Initialize the attribute description dataset.
        
        Args:
            name: Dataset name
            attribute_file: Path to attribute JSON file
            attributes_to_describe: List of attribute categories to describe
            normalize_attributes: Whether to normalize multi-value attributes
            no_object_id_input: Whether to hide object ID in instruction
            **kwargs: Other arguments for parent class
        """
        self.attribute_file = attribute_file
        self.normalize_attributes = normalize_attributes
        self.no_object_id_input = no_object_id_input
        
        # Default attribute categories
        self.attribute_categories = attributes_to_describe or [
            "color", "3D shape", "material", "usage", 
            "texture", "structure", "state"
        ]
        
        # Load attribute data
        self.attribute_data = self._load_attribute_data()
        
        # Override templates for this task
        kwargs["add_thinking_trace"] = False
        kwargs["parallel"] = False
        
        super().__init__(name=name, **kwargs)

    def _load_attribute_data(self) -> Dict:
        """Load attribute data from JSON file."""
        logger.info(f"Loading attribute data from {self.attribute_file}")
        with open(self.attribute_file, 'r') as f:
            data = json.load(f)
        
        # Normalize attributes if needed
        if self.normalize_attributes:
            data = self._normalize_attribute_data(data)
        
        total_objects = sum(len(objects) for objects in data.values())
        logger.info(f"Loaded attributes for {total_objects} objects across {len(data)} scenes")
        
        return data
    
    def _normalize_attribute_data(self, data: Dict) -> Dict:
        """
        Normalize multi-value attributes to use consistent separator.
        Converts all separators (/, and, ,) to "or".
        """
        logger.info("Normalizing attribute data...")
        
        def parse_and_join(value: str) -> str:
            """Parse multi-value string and rejoin with 'or'."""
            if not value or not isinstance(value, str):
                return value
            
            # Split by various separators
            pattern = r'\s*[/,]\s*|\s+and\s+|\s+or\s+'
            values = re.split(pattern, value)
            
            # Clean and filter
            cleaned = [v.lower().strip() for v in values if v.strip()]
            
            # Rejoin with "or"
            return " or ".join(cleaned) if len(cleaned) > 1 else cleaned[0] if cleaned else ""
        
        normalized_data = {}
        for scene_id, objects in data.items():
            normalized_data[scene_id] = {}
            for obj_id, attributes in objects.items():
                normalized_data[scene_id][obj_id] = {}
                for attr_name, attr_value in attributes.items():
                    if isinstance(attr_value, str):
                        normalized_data[scene_id][obj_id][attr_name] = parse_and_join(attr_value)
                    elif isinstance(attr_value, list):
                        # Handle list values
                        normalized_values = [parse_and_join(v) if isinstance(v, str) else v for v in attr_value]
                        normalized_data[scene_id][obj_id][attr_name] = " or ".join(normalized_values)
                    else:
                        normalized_data[scene_id][obj_id][attr_name] = attr_value
        
        return normalized_data
    
    def _initialize_templates(self, **kwargs):
        """Initialize templates for single-object attribute description."""
        # Single object attribute description templates
        if self.no_object_id_input:
            kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
                "Describe the attributes of the object in the scene. Provide the following attributes: {attributes}.\n",
                "For the object in this scene, describe these attributes: {attributes}.\n",
                "What are the {attributes} of the object?\n",
                "Describe the object's {attributes}.\n",
            ])
        else:
            kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
                "Describe the attributes of object with ID {object_id}. Provide the following attributes: {attributes}.\n",
                "For object {object_id}, describe these attributes: {attributes}.\n",
                "What are the {attributes} of object {object_id}?\n",
                "Tell me about object {object_id}'s {attributes}.\n",
            ])

        # we need to call this to set up other templates, and prepend thinking + object set templates before instruction templates
        super()._initialize_templates(**kwargs)

        
        # Response templates for single object
        self.response_templates = [
            "Roger. Object {id}({object_name}) has these attributes: {attributes}",
            "Apeiria found object {id}({object_name}). Attributes: {attributes}",
            "Object {id} is a {object_name}. {attributes}",
            "Here are the attributes for object {id}({object_name}): {attributes}",
        ]
        
        # Template for single attribute
        self.single_attr_templates = [
            "{attr_name}: {attr_value}",
            "{attr_name} is {attr_value}",
            "its {attr_name} is {attr_value}",
        ]

        # Fix templates if not training or fix_template is set
        if self.split != "train" or self.fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
            self.single_attr_templates = [self.single_attr_templates[0]]
    
    def _load_annotations(self):
        """Return empty list as we generate samples from scene data."""
        return []
    
    def _generate_samples(self, load_from_cache):
        """
        Generate one sample per object with attributes.
        Similar to Real3DObjectInfoDataset but for attributes.
        """
        logger.info(f"Generating attribute samples (one per object)...")
        
        samples = []
        
        for scene_id in self.scene_list:
            if scene_id not in self.scene_data:
                continue
            
            # Check if this scene has attribute data
            if scene_id not in self.attribute_data:
                logger.debug(f"No attribute data for scene {scene_id}")
                continue
            
            # Get objects in this scene
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            object_ids = instance_bboxes[:, 7].astype(int)
            object_classes = instance_bboxes[:, 6].astype(int)
            
            # For each object with attribute data
            for idx, obj_id in enumerate(object_ids):
                obj_id_str = str(obj_id)
                
                # Check if this object has attribute data
                if obj_id_str not in self.attribute_data[scene_id]:
                    continue
                
                # Get object info
                object_name = self.object_classes[int(object_classes[idx])]
                obj_attrs = self.attribute_data[scene_id][obj_id_str]
                
                # Format attributes
                attr_strings = []
                for attr_name in self.attribute_categories:
                    if attr_name in obj_attrs and obj_attrs[attr_name]:
                        attr_value = obj_attrs[attr_name]
                        attr_str = random.choice(self.single_attr_templates).format(
                            attr_name=attr_name,
                            attr_value=attr_value
                        )
                        attr_strings.append(attr_str)
                
                # Skip if no valid attributes
                if not attr_strings:
                    continue
                
                # Create instruction
                attr_list = ", ".join(self.attribute_categories)
                if self.no_object_id_input:
                    description = random.choice(self.instruction_templates).format(
                        attributes=attr_list
                    )
                else:
                    description = random.choice(self.instruction_templates).format(
                        object_id=obj_id,
                        attributes=attr_list
                    )
                
                # Create expected response
                combined_attrs = ", ".join(attr_strings)
                expected_response = random.choice(self.response_templates).format(
                    id=obj_id,
                    object_name=object_name,
                    attributes=combined_attrs
                )
                
                # Create sample ID(s)
                ann_id = len(samples)
                scanrefer_id = f"{scene_id}|{ann_id}"
                hash_id = f"real_attr_{scene_id}_{obj_id}"
                scan2cap_id = f"{scene_id}|{obj_id}|{object_name}"
                
                samples.append({
                    "scene_id": scene_id,
                    "ann_id": str(ann_id),
                    "description": description,
                    "program": f"get_object_attributes(scene(), {obj_id})",
                    "object_name": object_name,
                    "object_id": int(obj_id),
                    "object_ids": [int(obj_id)],
                    "scanrefer_id": scanrefer_id,
                    "hash_id": hash_id,
                    "expected_response": expected_response,
                    "question_id": scanrefer_id,
                    "raw_question_id": scanrefer_id,
                    "scan2cap_id": scan2cap_id,
                    "data_type": self.name,
                    "split": self.split,
                    "target_obj_idx": idx,  # Index in instance_bboxes
                })
        
        logger.info(f"Generated {len(samples)} attribute samples (one per object)")
        return samples
    
    def __getitem__(self, idx):
        """
        Get a single sample, similar to Real3DObjectInfoDataset.
        Only the target object should be visible via mask.
        """
        self.accessed_times[idx] += 1
        
        # Get sample data
        data = self.samples[idx]
        scene_id = data["scene_id"]
        target_obj_idx = data["target_obj_idx"]
        
        # Get scene data
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()
        target_bbox = instance_bboxes[target_obj_idx, 0:6].copy()
        
        # Get object features
        all_object_features = self.frozen_features[scene_id][0]
        all_object_mask = self.frozen_features[scene_id][1]
        all_predicted_bbox_corners = self.frozen_features[scene_id][2]
        all_input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        all_object_labels = instance_bboxes[:, 6].astype(np.int64)
        
        # Create a mask that only includes this object
        single_object_mask = torch.zeros_like(all_object_mask)
        single_object_mask[target_obj_idx] = 1
        
        # Create PC dictionary with just this object
        pc_dict = {
            "object_feature": all_object_features,
            "object_mask": single_object_mask,  # Only this object is visible
            "predicted_bbox_corners": all_predicted_bbox_corners,
            "input_predicted_bbox": all_input_predicted_bbox,
            "object_labels": all_object_labels,
            "object_ious": np.ones(len(all_object_features), dtype=np.float32),
        }

        if self.image_encoder_name:
            image_ids = self._get_image_indices(data)
            pc_dict["image_embeds"] = self._get_image_features(scene_id, image_ids) # [N_views, L, D] or [N_views, 1, D]
        
        return {
            "question_id": data["question_id"],
            "raw_question_id": data["raw_question_id"],
            "scan2cap_id": data["scan2cap_id"],
            "scene_id": scene_id,
            "scanrefer_id": data["scanrefer_id"],
            "hash_id": data["hash_id"],
            "target_id": target_obj_idx,
            "target_pred_id": target_obj_idx,
            "object_id": data["object_id"],
            "object_ids": data["object_ids"],
            "data_type": self.name,
            "split": self.split,
            "target_bbox": target_bbox,
            "program": data["program"],
            "program_complexity": 1,
            "description": data["description"],
            "expected_response": data["expected_response"],
            "shuffle_indices": np.arange(len(all_object_features)),
            "revert_indices": np.arange(len(all_object_features)),
            **pc_dict,
        }
    
    def evaluate(self, preds: Dict[str, str], gt_indices: Dict, 
                 hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Evaluate attribute descriptions for single objects.
        Uses attribute-level precision, recall, and F1.
        """
        logger.info("Evaluating single-object attribute descriptions...")
        
        # Metrics per attribute category
        attr_metrics = {attr_name: {
            'tp': 0,  # true positives
            'fp': 0,  # false positives
            'fn': 0,  # false negatives
            'total_pred': 0,
            'total_gt': 0,
        } for attr_name in self.attribute_categories}
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_samples = 0
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            scene_id = sample["scene_id"]
            obj_id_str = str(sample["object_id"])
            
            # Get ground truth attributes
            if scene_id not in self.attribute_data or obj_id_str not in self.attribute_data[scene_id]:
                logger.warning(f"No GT attributes for {scene_id}/{obj_id_str}")
                continue
            
            gt_obj_attrs = self.attribute_data[scene_id][obj_id_str] # Dict of attr_name -> value
            
            # Parse predicted attributes
            pred_attrs = self._parse_single_object_attributes(pred_text.lower())
            
            total_samples += 1
            
            # Compare each attribute category
            for attr_name in self.attribute_categories:
                pred_values = set(pred_attrs.get(attr_name, []))
                
                # Parse ground truth values
                gt_value_str = gt_obj_attrs.get(attr_name, "")
                gt_values = set()
                if gt_value_str:
                    gt_values = set(v.strip().lower() for v in gt_value_str.split(" or ") if v.strip())
                
                # Update counts
                attr_metrics[attr_name]['total_pred'] += len(pred_values)
                attr_metrics[attr_name]['total_gt'] += len(gt_values)
                
                # Calculate matches
                matches = pred_values & gt_values
                tp = len(matches)
                fp = len(pred_values - gt_values)
                fn = len(gt_values - pred_values)
                
                attr_metrics[attr_name]['tp'] += tp
                attr_metrics[attr_name]['fp'] += fp
                attr_metrics[attr_name]['fn'] += fn
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
        
        # Calculate metrics
        def calc_metrics(tp, fp, fn):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return precision, recall, f1
        
        # Overall metrics
        overall_precision, overall_recall, overall_f1 = calc_metrics(total_tp, total_fp, total_fn)
        
        # Per-attribute metrics
        attr_results = {}
        for attr_name, counts in attr_metrics.items():
            precision, recall, f1 = calc_metrics(counts['tp'], counts['fp'], counts['fn'])
            attr_results[attr_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': counts['tp'],
                'fp': counts['fp'],
                'fn': counts['fn'],
            }
        
        # Build metrics dictionary
        metrics = {
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_evaluated': total_samples,
        }
        
        # Add per-attribute metrics
        for attr_name, attr_metric in attr_results.items():
            metrics[f'{attr_name}_precision'] = attr_metric['precision']
            metrics[f'{attr_name}_recall'] = attr_metric['recall']
            metrics[f'{attr_name}_f1'] = attr_metric['f1']
        
        # Build message
        message_lines = [
            f"Single-Object Attribute Evaluation ({total_samples} objects):",
            f"Overall: P={overall_precision:.4f}, R={overall_recall:.4f}, F1={overall_f1:.4f}",
            f"  TP={total_tp}, FP={total_fp}, FN={total_fn}",
            f"\nPer-Attribute Metrics:"
        ]
        
        for attr_name in self.attribute_categories:
            attr_metric = attr_results[attr_name]
            message_lines.append(
                f"  {attr_name:15s}: "
                f"P={attr_metric['precision']:.3f}, "
                f"R={attr_metric['recall']:.3f}, "
                f"F1={attr_metric['f1']:.3f} "
                f"(TP={attr_metric['tp']}, FP={attr_metric['fp']}, FN={attr_metric['fn']})"
            )
        
        message = "\n".join(message_lines)
        return message, metrics

    def _parse_single_object_attributes(self, response_text: str) -> Dict[str, List[str]]:
        """
        Parse attributes from a single-object response.
        
        Args:
            response_text: Model's response text (lowercased)
            
        Returns:
            Dictionary mapping attribute_name -> list of values
        """
        obj_attrs = {}
        
        # Try to find the attributes section
        # Expected format: "Object X(...): attr1: value1, attr2: value2" or similar
        
        # First, try to extract everything after ":" or "attributes:"
        attr_text = response_text
        if "attributes:" in response_text:
            attr_text = response_text.split("attributes:", 1)[1]
        elif ")" in response_text and ":" in response_text:
            # Format like "Object 1(chair): ..."
            parts = response_text.split(")", 1)
            if len(parts) > 1 and ":" in parts[1]:
                attr_text = parts[1].split(":", 1)[1]
        elif "." in response_text:
            # Format like "Object 1 is a chair. attr1: value1, ..."
            parts = response_text.split(".", 1)
            if len(parts) > 1:
                attr_text = parts[1]
        else:
            logger.warning(f"Could not find attributes section in response: ||{response_text}||")
        
        # Split by commas carefully (handle "or" inside values)
        attr_parts = []
        current_part = ""
        paren_depth = 0
        
        for char in attr_text:
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
            elif char == ',' and paren_depth == 0:
                attr_parts.append(current_part.strip())
                current_part = ""
                continue
            current_part += char
        
        if current_part.strip():
            attr_parts.append(current_part.strip())
        
        # Parse each attribute part
        for part in attr_parts:
            # Match "attr_name: value" or "attr_name is value" or "its attr_name is value"
            part = part.replace("its ", "").strip()
            attr_match = re.match(r'([^:]+?)(?:\s+is\s+|\s*:\s*)(.+)', part, re.IGNORECASE)
            if not attr_match:
                continue
            
            attr_name = attr_match.group(1).strip().lower()
            attr_value = attr_match.group(2).strip().lower()
            
            # Match to our attribute categories
            matched_category = None
            for category in self.attribute_categories:
                if category.lower() in attr_name or attr_name in category.lower():
                    matched_category = category
                    break
            
            if not matched_category:
                continue
            
            # Split values by "or"
            values = [v.strip() for v in attr_value.split(" or ") if v.strip()]
            obj_attrs[matched_category] = values
        
        return obj_attrs
    
    def _save_samples(self):
        """Disable saving."""
        pass
    
    def _load_samples(self):
        """Disable loading from cache."""
        pass

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)

class Real3DDenseCaptioningDataset(Real3DDataset):
    """
    Dataset for dense captioning task - one sample per object per caption.
    Each object may have multiple samples if it has multiple captions.
    Uses captions from SR3D/NR3D/ScanRefer annotations.
    
    Unlike ObjectInfo/Attributes datasets, this uses ALL object embeddings in the scene,
    as the model needs context from surrounding objects for better captioning.
    
    For evaluation (non-train splits), each object has only ONE sample for inference,
    but evaluation compares against ALL ground truth captions for that object.
    """
    
    def __init__(
        self,
        name: str = "sr3d_dense",  # or "nr3d_dense", "scanrefer_dense"
        max_captions_per_object: int = None,  # Limit captions per object
        shuffle_captions: bool = False,  # Shuffle captions for each object. NOTE: better leave for dataloader for randomness
        add_dummy_caption: bool = True,  # Add dummy caption for unmatched GT boxes in proposal eval
        **kwargs
    ):
        """
        Initialize the dense captioning dataset.
        
        Args:
            name: Dataset name (should include "dense" suffix)
            max_captions_per_object: Maximum captions to use per object (None = use all)
            shuffle_captions: Whether to shuffle captions order
            add_dummy_caption: Whether to add dummy captions for unmatched GT boxes
            **kwargs: Other arguments for parent class
        """
        self.max_captions_per_object = max_captions_per_object
        self.no_object_id_input = False # Dense captioning always includes object ID to let model know which object to describe
        self.shuffle_captions = shuffle_captions
        self.dataset_name = name.replace("_dense", "")
        self.add_dummy_caption = add_dummy_caption
        
        # Override templates for this task
        kwargs["add_thinking_trace"] = kwargs.get("add_thinking_trace", False) # TODO: add CoT to delibrately consider related objects?
        kwargs["parallel"] = False
        
        super().__init__(name=name, **kwargs)

        # Build corpus for proposal-based evaluation
        self._build_corpus()

    def _build_corpus(self):
        """
        Build GT caption corpus organized by scene_id -> object_id -> list of captions.
        Used for proposal-based evaluation.
        """
        self.corpus = {}  # key format: "scene_id|object_id|object_name" -> list of captions
        
        for scene_id, scene_captions in self.dense_captions.items():
            if scene_id not in self.scene_data:
                continue
                
            # Get object info for this scene
            # instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            # get from GT
            gt_objects = self.scene_data[scene_id]["objects"]
            gt_boxes = np.array([obj.get("location", [0, 0, 0]) + obj.get("size", [0,0,0]) for obj in gt_objects])
            gt_ids = np.array([obj["id"] for obj in gt_objects])
            gt_class_ids = np.array([self.object_label_type.name_to_id[obj["name"]] for obj in gt_objects])

            for obj_id_str, captions in scene_captions.items():
                if len(captions) == 0:
                    continue

                obj_id = int(obj_id_str)
                
                # Find object name
                object_name = "unknown"
                for idx, oid in enumerate(gt_ids):
                    if oid == obj_id:
                        object_name = self.object_classes[int(gt_class_ids[idx])]
                        if object_name in ["wall", "floor", "ceiling"]:
                            logger.warning(f"Object shall not be wall/floor/ceiling: {scene_id}|{obj_id}")
                        break
                
                # Create corpus key (same format as scan2cap_id)
                corpus_key = f"{scene_id}|{obj_id}|{object_name}"
                self.corpus[corpus_key] = captions
        
        total_captions = sum(len(caps) for caps in self.corpus.values())
        logger.info(f"Built corpus with {total_captions} captions for {len(self.corpus)} objects")
    
    def _should_use_proposal_eval(self) -> bool:
        """
        Determine if proposal-based evaluation should be used.
        Returns True if using proposal boxes and not in training mode.
        """
        return (
            hasattr(self, 'use_proposal_box_as_input') and 
            self.use_proposal_box_as_input and 
            self.split != "train"
        )

    def process_predictions_with_proposals(
        self,
        predictions: Dict[str, str],
        iou_threshold: float = 0.5,
        dummy_caption: str = "sos eos",
        method: str = "recall",
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Process predictions for proposal-based evaluation.
        Matches predicted boxes to GT boxes using IoU threshold.
        
        Args:
            predictions: Dict mapping scanrefer_id to predicted caption text
            iou_threshold: IoU threshold for matching boxes
            dummy_caption: Caption to use for unmatched GT boxes
            method: "recall" or "precision" evaluation method
            
        Returns:
            gt: Dict mapping "scene_id|object_id" to list of GT captions
            pred: Dict mapping "scene_id|object_id" to list of predicted captions
        """
        logger.critical(f"Received {len(predictions)} predictions, {len(self.corpus)} GT objects in corpus.")

        # Organize GT corpus by scene_id -> object_id -> captions
        gt_scene_id_to_object_id_to_caption = defaultdict(lambda: defaultdict(list))
        for key, captions in self.corpus.items():
            scene_id, object_id, object_name = key.split("|")
            gt_scene_id_to_object_id_to_caption[scene_id][object_id] = list(captions)
        
        # Organize predictions by scene_id -> pred_object_id -> caption
        predictions_corpus = defaultdict(dict)
        for key, caption in predictions.items():
            # Extract caption from response format
            pred_caption = caption
            pred_caption = self._extract_caption_from_prediction(pred_caption)
            
            # Get sample info
            if key not in self.scanrefer_id_to_idx:
                continue
            sample = self.samples[self.scanrefer_id_to_idx[key]]
            scene_id = sample["scene_id"]
            pred_object_id = str(sample["object_id"])  # This is the predicted box ID
            
            predictions_corpus[scene_id][pred_object_id] = pred_caption
        
        pred_scene_id_to_object_id_to_caption = defaultdict(dict)
        
        if method == "recall":
            # For each GT box, find best matching predicted box
            logger.info("Processing predictions with recall method...")
            accepted_iou, used_gts = 0, 0
            
            for scene_id in predictions_corpus.keys():
                if scene_id not in self.scene_data:
                    continue
                
                # Get predicted boxes (from proposal features)
                pred_bboxes = self.input_predicted_bboxes[scene_id]  # (K1, 6)
                
                # Get GT boxes from scene data
                # gt_objects = self.scene_data[scene_id]["original_objects"]
                gt_objects = self.scene_data[scene_id]["objects"]
                gt_bboxes = np.array([
                    obj["location"] + obj["size"] for obj in gt_objects
                ])  # (K2, 6)
                gt_object_ids = np.array([obj["id"] for obj in gt_objects])  # (K2,)
                
                # Calculate IoU matrix
                iou_matrix = mutual_iou_vectorized(pred_bboxes, gt_bboxes)  # (K1, K2)

                if iou_matrix.size > 0:
                    pass
                    # logger.info(f"Mean Max Recall IoU for this: {scene_id}: {np.mean(iou_matrix.max(axis=0)):.4f}")
                    # logger.info(f"Mean Max Recall IoU > {iou_threshold} for this: {scene_id}: {np.mean((iou_matrix.max(axis=0) >= iou_threshold).astype(np.float32)):.4f}")
                
                # For each GT box, find best matching pred box
                for i, gt_object_id in enumerate(gt_object_ids):
                    gt_object_id_str = str(int(gt_object_id))
                    
                    if gt_object_id_str not in gt_scene_id_to_object_id_to_caption[scene_id]:
                        continue  # Skip GT objects without captions
                    
                    used_gts += 1
                    
                    # Find best matching predicted box
                    if iou_matrix.shape[0] > 0:
                        best_pred_idx = iou_matrix[:, i].argmax()
                        best_iou = iou_matrix[best_pred_idx, i]
                    else:
                        best_iou = 0
                    
                    if best_iou >= iou_threshold:
                        # Use the prediction for this GT box
                        pred_id_str = str(best_pred_idx)
                        if pred_id_str in predictions_corpus[scene_id]:
                            pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id_str] = (
                                predictions_corpus[scene_id][pred_id_str]
                            )
                            accepted_iou += 1
                        else:
                            logger.critical(f"Prediction for scene {scene_id} pred_id {pred_id_str} not found, which ALL predicted boxes shall make a caption.")
                            logger.critical(f"All existing predictions: {list(predictions_corpus[scene_id].keys())}")
            
            accepted_rate = accepted_iou / used_gts if used_gts > 0 else 0
            logger.info(f"Accepted bbox@{iou_threshold}: {accepted_iou}/{used_gts}={accepted_rate:.4f}")
            
        elif method == "precision":
            # For each predicted box, find best matching GT box
            logger.info("Processing predictions with precision method...")
            accepted_iou, used_preds = 0, 0
            
            for scene_id in gt_scene_id_to_object_id_to_caption.keys():
                if scene_id not in self.scene_data:
                    continue
                if scene_id not in predictions_corpus:
                    continue
                
                # Get predicted and GT boxes
                pred_bboxes = self.input_predicted_bboxes[scene_id]
                gt_objects = self.scene_data[scene_id]["objects"]
                gt_bboxes = np.array([
                    obj["location"] + obj["size"] for obj in gt_objects
                ])
                gt_object_ids = np.array([obj["id"] for obj in gt_objects])
                
                # Calculate IoU matrix
                iou_matrix = mutual_iou_vectorized(pred_bboxes, gt_bboxes)
                
                # For each predicted box
                for pred_idx in range(len(pred_bboxes)):
                    pred_id_str = str(pred_idx)
                    if pred_id_str not in predictions_corpus[scene_id]:
                        continue
                    
                    # Find best matching GT box
                    if iou_matrix.shape[1] > 0:
                        best_gt_idx = iou_matrix[pred_idx, :].argmax()
                        best_iou = iou_matrix[pred_idx, best_gt_idx]
                        gt_object_id_str = str(int(gt_object_ids[best_gt_idx]))
                    else:
                        best_iou = 0
                        gt_object_id_str = None
                    
                    if gt_object_id_str not in gt_scene_id_to_object_id_to_caption[scene_id]:
                        continue
                    
                    used_preds += 1
                    
                    if best_iou >= iou_threshold:
                        pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id_str] = (
                            predictions_corpus[scene_id][pred_id_str]
                        )
                        accepted_iou += 1
                    else:
                        # Add dummy caption for unmatched prediction
                        pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id_str] = dummy_caption
            
            accepted_rate = accepted_iou / used_preds if used_preds > 0 else 0
            logger.info(f"Accepted bbox@{iou_threshold}: {accepted_iou}/{used_preds}={accepted_rate:.4f}")
        
        else:
            raise ValueError(f"Unknown evaluation method: {method}")
        
        # Flatten to final format
        pred = self._flatten_corpus(pred_scene_id_to_object_id_to_caption)
        gt = self._flatten_corpus(gt_scene_id_to_object_id_to_caption)
        
        # Add dummy captions for unmatched GT boxes (recall method)
        if self.add_dummy_caption and method == "recall":
            for k in gt.keys():
                if k not in pred:
                    pred[k] = [dummy_caption]
        
        # Remove unmatched predictions (precision method)
        if method == "precision":
            pred = {k: pred[k] for k in gt.keys() if k in pred}
        
        # Show some examples
        if len(pred) > 0 and len(gt) > 0:
            common_keys = list(set(pred.keys()).intersection(set(gt.keys())))
            n_show = min(3, len(common_keys))
            if n_show > 0:
                show_keys = np.random.choice(common_keys, n_show, replace=False)
                logger.info(f"Showing {n_show} examples:")
                for key in show_keys:
                    logger.info(f"GT: {key}: {gt[key][:2]}...")  # Show first 2 GT captions
                    logger.info(f"Pred: {key}: {pred[key]}")
        
        return gt, pred
    
    def _flatten_corpus(self, corpus: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Flatten corpus from scene_id -> object_id -> caption(s) to "scene_id|object_id" -> caption(s).
        """
        new_corpus = {}
        for scene_id, object_id_to_caption in corpus.items():
            for object_id, captions in object_id_to_caption.items():
                key = f"{scene_id}|{object_id}"
                if isinstance(captions, list):
                    new_corpus[key] = captions
                else:
                    new_corpus[key] = [captions]
        return new_corpus
    
    def _initialize_templates(self, **kwargs):
        """Initialize templates for single-object dense captioning."""
        # Set instruction templates BEFORE calling super()
        # so that super() can prepend thinking + |object_set| to them
        kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
            f"Describe object with ID {{object_id}} in {self.dataset_name.upper()} style.\n",
            f"In {self.dataset_name.upper()} style, what is object {{object_id}}? Please describe it.\n",
            f"Provide a description of object {{object_id}} in {self.dataset_name.upper()} style.\n",
            f"In {self.dataset_name.upper()} style, tell me about object {{object_id}} .\n",
            f"Give me a detailed description of object {{object_id}}, in {self.dataset_name.upper()} style.\n",
        ])
        
        # Call super() to add common prefixes (thinking prompt + |object_set|)
        super()._initialize_templates(**kwargs)
        
        # Response templates for single object with caption
        self.response_templates = [
            "Roger. Object {id}({object_name}): {caption}",
            "Apeiria found object {id}({object_name}). {caption}",
            "Object {id} is a {object_name}. {caption}",
            "Here's the description for object {id}({object_name}): {caption}",
            "Object {id}({object_name}) - {caption}",
        ]
        
        # Fix templates if not training or fix_template is set
        if self.split != "train" or self.fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
    
    def _load_annotations(self):
        """
        Load original annotations and collect captions for dense captioning.
        """
        # Get the annotation file from parent class
        annotation_file = self._get_annotation_file(self.split)
        logger.info(f"Loading dense caption annotations from {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            raw_annotations = json.load(f)
        
        # Collect all captions for each object
        self.dense_captions = self._collect_dense_captions(raw_annotations)
        
        total_objects = sum(len(objs) for objs in self.dense_captions.values())
        total_captions = sum(
            len(captions) 
            for scene_objs in self.dense_captions.values() 
            for captions in scene_objs.values()
        )
        logger.info(
            f"[{self.name}] Collected {total_captions} captions for "
            f"{total_objects} unique objects across {len(self.dense_captions)} scenes"
        )
        
        return []  # Return empty as we'll generate samples in _generate_samples
    
    def _collect_dense_captions(self, annotations: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
        """
        Collect all captions for each object from annotations.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Dictionary: scene_id -> object_id (str) -> list of captions
        """
        captions = defaultdict(lambda: defaultdict(list))
        
        for anno in annotations:
            scene_id = anno["scene_id"]
            object_id = str(anno["object_id"])
            description = anno["description"].strip()
            
            if scene_id and object_id and description:
                captions[scene_id][object_id].append(description)
        
        # Convert to regular dict and optionally limit/shuffle captions
        result = {}
        for scene_id, scene_objs in captions.items():
            result[scene_id] = {}
            for obj_id, caption_list in scene_objs.items():
                # Remove duplicates while preserving order
                seen = set()
                unique_captions = []
                for cap in caption_list:
                    if cap not in seen:
                        seen.add(cap)
                        unique_captions.append(cap)
                
                # Shuffle if needed
                if self.shuffle_captions and self.split == "train":
                    random.shuffle(unique_captions)
                
                # Limit if needed
                if self.max_captions_per_object is not None:
                    unique_captions = unique_captions[:self.max_captions_per_object]
                
                result[scene_id][obj_id] = unique_captions
        
        return result
    
    def _generate_samples(self, load_from_cache):
        """
        Generate samples based on split:
        - Training: one sample per (object, caption) pair
        - Evaluation: ONE sample per object, but keep all captions for comparison
        """
        logger.info(f"Generating dense captioning samples...")
        
        samples = []
        # Store all captions for each object (for evaluation with all references)
        self.object_all_captions = {}
        
        for scene_id in self.scene_list:
            if scene_id not in self.scene_data:
                continue
            
            # Check if this scene has captions
            if scene_id not in self.dense_captions:
                logger.debug(f"No captions found for scene {scene_id}")
                continue
            
            # Get objects in this scene
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            object_ids = instance_bboxes[:, 7].astype(int)
            object_classes = instance_bboxes[:, 6].astype(int)
            
            # For each object with captions
            for idx, obj_id in enumerate(object_ids):
                obj_id_str = str(obj_id)
                
                # Check if this object has captions
                if obj_id_str not in self.dense_captions[scene_id]:
                    if self.split != "train":
                        # this is especially possible, for proposal-based eval, since dense_captions use GT id, 
                        # while predicted proposal use completely non-corresponding ids, using it to retrieve captions will likely get nothing.
                        obj_captions = [""]
                    else: 
                        continue
                else:
                    # Get all captions for this object
                    obj_captions = self.dense_captions[scene_id][obj_id_str]
                
                # if not obj_captions:
                #     continue
                
                # Create object key for looking up all captions during evaluation
                obj_key = f"{scene_id}|{obj_id}"
                self.object_all_captions[obj_key] = obj_captions
                
                # Get object info
                object_name = self.object_classes[int(object_classes[idx])]
                
                # Determine how many samples to create
                if self.split != "train":
                    # Evaluation: only ONE sample per object (using first caption)
                    captions_to_use = [obj_captions[0]] # completely dummy: they are never correponding to this object
                else:
                    # Training: one sample per caption
                    captions_to_use = obj_captions
                
                # Create samples
                for caption_idx, caption in enumerate(captions_to_use):
                    # Create instruction
                    description = random.choice(self.instruction_templates).format(object_id=obj_id)
                    
                    # Create expected response
                    expected_response = random.choice(self.response_templates).format(
                        id=obj_id,
                        object_name=object_name,
                        caption=caption
                    )
                    
                    # Create unique sample IDs
                    ann_id = len(samples)
                    scanrefer_id = f"{scene_id}|{ann_id}"
                    hash_id = f"real_dense_{scene_id}_{obj_id}_cap{caption_idx}"
                    scan2cap_id = f"{scene_id}|{obj_id}|{object_name}"
                    
                    samples.append({
                        "scene_id": scene_id,
                        "ann_id": str(ann_id),
                        "description": description,
                        "raw_description": caption,  # The caption used for this sample
                        "program": f"describe_object(scene(), {obj_id})",
                        "object_name": object_name,
                        "object_id": int(obj_id),
                        "object_ids": [int(obj_id)],
                        "caption_idx": caption_idx,
                        "total_captions": len(obj_captions),  # Total captions for this object
                        "scanrefer_id": scanrefer_id,
                        "hash_id": hash_id,
                        "expected_response": expected_response,
                        "question_id": scanrefer_id,
                        "raw_question_id": f"{scene_id}_{obj_id}_{caption_idx}",
                        "scan2cap_id": scan2cap_id,
                        "data_type": self.name,
                        "split": self.split,
                        "target_obj_idx": idx,  # Index in instance_bboxes
                        "obj_key": obj_key,  # Key for looking up all GT captions
                    })
        
        # Log statistics
        total_all_captions = sum(len(caps) for caps in self.object_all_captions.values())
        
        if self.split != "train":
            logger.info(
                f"Generated {len(samples)} dense captioning samples for EVALUATION "
                f"(one per object, will compare with all {total_all_captions} GT captions)"
            )
        else:
            logger.info(
                f"Generated {len(samples)} dense captioning samples for TRAINING "
                f"(one per object per caption)"
            )
            
            objects_with_multiple_captions = sum(
                1 for sample in samples 
                if sample["total_captions"] > 1
            )
            avg_captions = sum(s["total_captions"] for s in samples) / len(samples) if samples else 0
            logger.info(
                f"Training statistics: {objects_with_multiple_captions} objects have multiple captions, "
                f"avg {avg_captions:.2f} captions per object"
            )
        
        return samples
    
    def evaluate_c2c(self, preds: Dict[str, str], gt_indices: Dict, 
                 hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Caption-to-caption evaluation.
        Each prediction is compared with its corresponding single GT caption.
        (This method is mainly for debugging/comparison purposes)
        """
        logger.info("Evaluating dense captions with caption-to-caption matching...")
        
        candidates = {}
        corpus = {}
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                # Convert hash_id to scanrefer_id
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            # Extract the caption part from prediction
            pred_caption = pred_text
            # caption_match = re.search(
            #     r'(?:object\s+\d+\s*\([^)]+\)\s*[:\-]\s*|object\s+\d+\s*[:\-]\s*)(.+)',
            #     pred_text,
            #     re.IGNORECASE | re.DOTALL
            # )
            caption_match = re.search(
                r'(?:object\s+\d+\s*\([^)]+\)\s*[:\-\.]\s*|'  # Object X(name)[: - .] 
                r'object\s+\d+\s+is\s+a\s+[^.]+\.\s*|'        # Object X is a name.
                r'object\s+\d+\s*[:\-\.]\s*)'                 # Object X[: - .]
                r'(.+)',
                pred_text,
                re.IGNORECASE | re.DOTALL
            )
            if caption_match:
                pred_caption = caption_match.group(1).strip()
            
            # Get the single GT caption for this sample
            gt_caption = sample["raw_description"]
            
            candidates[scanrefer_id] = [pred_caption] 
            corpus[scanrefer_id] = [gt_caption]  # Single GT caption
        
        if not candidates:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}

        # normalize all captions
        for k in corpus:
            corpus[k] = [postprocess_punctuation_for_caption_metrics(preprocess_sos_eos_for_scan2cap(cap)) for cap in corpus[k]]
        for k in candidates:
            candidates[k] = [postprocess_punctuation_for_caption_metrics(preprocess_sos_eos_for_scan2cap(cap)) for cap in candidates[k]]

        # Use eval_utils to calculate scores
        score_per_caption, message, metrics = score_captions(corpus, candidates)
        metrics["total_evaluated"] = len(candidates)
        
        full_message = (
            f"Dense Captioning C2C Evaluation ({len(candidates)} captions):\n"
            f"{message}"
        )
        
        return full_message, metrics
    

    def evaluate(self, preds: Dict[str, str], gt_indices: Dict,
                 hash_id_index: bool = False,
                 iou_threshold: List[float] = [0.25, 0.5],
                 iou_thresholds: List[float] = [0.25, 0.5],
                 use_proposal_eval: bool = None,
                 method: str = "recall",
                 dummy_caption: str = "sos eos",
                 **kwargs) -> Tuple[str, Dict]:
        """
        Evaluate dense captioning predictions.
        
        Args:
            preds: Dict mapping scanrefer_id to predicted caption text
            gt_indices: Not used, kept for API compatibility
            hash_id_index: Whether keys in preds are hash_ids
            iou_threshold: IoU threshold for proposal-based evaluation
            use_proposal_eval: Force proposal evaluation mode (None = auto-detect)
            method: "recall" or "precision" for proposal evaluation
            dummy_caption: Caption for unmatched GT boxes
            
        Returns:
            message: Evaluation summary
            metrics: Dict of evaluation metrics
        """
        # Determine evaluation mode
        if use_proposal_eval is None:
            use_proposal_eval = self._should_use_proposal_eval()
        
        if use_proposal_eval:
            logger.info(f"Using proposal-based evaluation with IoU threshold {iou_threshold}")
            all_messages = []
            all_metrics = {}
            for thr in iou_thresholds:
                message, metrics = self._evaluate_with_proposals(
                    preds, hash_id_index, thr, method, dummy_caption
                )
                # make metrics "_iou@{thr}"
                metrics = {f"{k}@{thr}": v for k, v in metrics.items()}
                all_messages.append(message)
                all_metrics.update(metrics)
            full_message = "\n\n".join(all_messages)
            return full_message, all_metrics
        else:
            logger.info("Using standard GT-based evaluation")
            return self._evaluate_standard(preds, hash_id_index)
    
    def _evaluate_with_proposals(
        self,
        preds: Dict[str, str],
        hash_id_index: bool,
        iou_threshold: float,
        method: str,
        dummy_caption: str,
    ) -> Tuple[str, Dict]:
        """
        Proposal-based evaluation for dense captioning.
        """
        # Convert hash_ids to scanrefer_ids if needed
        if hash_id_index:
            converted_preds = {}
            for key, pred in preds.items():
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        converted_preds[sample["scanrefer_id"]] = pred
                        break
            preds = converted_preds
        
        # Process predictions with proposal matching
        gt, pred = self.process_predictions_with_proposals(
            preds, iou_threshold, dummy_caption, method
        )
        
        if not pred:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}
        
        # Normalize captions
        for k in gt:
            gt[k] = [
                postprocess_punctuation_for_caption_metrics(
                    preprocess_sos_eos_for_scan2cap(cap)
                ) for cap in gt[k]
            ]
        for k in pred:
            pred[k] = [
                postprocess_punctuation_for_caption_metrics(
                    preprocess_sos_eos_for_scan2cap(cap)
                ) for cap in pred[k]
            ]
        
        # Calculate caption metrics
        score_per_caption, message, metrics = score_captions(gt, pred)
        
        # Add proposal-specific metrics
        metrics["iou_threshold"] = iou_threshold
        metrics["evaluation_method"] = method
        metrics["total_gt_objects"] = len(gt)
        metrics["total_matched_objects"] = len([k for k in pred if pred[k][0] != dummy_caption])
        metrics["match_rate"] = metrics["total_matched_objects"] / len(gt) if len(gt) > 0 else 0
        
        full_message = (
            f"Dense Captioning Evaluation (Proposal-based, {method}):\n"
            f"  IoU Threshold: {iou_threshold}\n"
            f"  GT Objects: {metrics['total_gt_objects']}\n"
            f"  Matched Objects: {metrics['total_matched_objects']}\n"
            f"  Match Rate: {metrics['match_rate']:.4f}\n"
            f"\n{message}"
        )
        
        return full_message, metrics
    
    def _extract_caption_from_prediction(self, pred_text: str) -> str:
        """
        Extract the caption part from the prediction text.
        """
        caption_match = re.search(
            r'(?:object\s+\d+\s*\([^)]+\)\s*[:\-\.]\s*|'  # Object X(name)[: - .] 
            r'object\s+\d+\s+is\s+a\s+[^.]+\.\s*|'        # Object X is a name.
            r'object\s+\d+\s*[:\-\.]\s*)'                 # Object X[: - .]
            r'(.+)',
            pred_text,
            re.IGNORECASE | re.DOTALL
        )
        if caption_match:
            return caption_match.group(1).strip()
        return pred_text.strip()

    def _evaluate_standard(
        self,
        preds: Dict[str, str],
        hash_id_index: bool,
    ) -> Tuple[str, Dict]:
        """
        Standard GT-based evaluation (original evaluate method logic).
        Each prediction is compared against ALL ground truth captions for that object.
        """
        logger.info("Evaluating dense captions per object (using all GT captions as references)...")
        
        candidates = {}
        corpus = {}
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            # Extract prediction caption
            pred_caption = pred_text
            pred_caption = self._extract_caption_from_prediction(pred_text)
            
            # Get ALL GT captions for this object
            obj_key = sample["obj_key"]
            all_gt_captions = self.object_all_captions[obj_key]
            
            candidates[scanrefer_id] = [pred_caption]
            corpus[scanrefer_id] = all_gt_captions
        
        if not candidates:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}
        
        # Normalize captions
        for k in corpus:
            corpus[k] = [
                postprocess_punctuation_for_caption_metrics(
                    preprocess_sos_eos_for_scan2cap(cap)
                ) for cap in corpus[k]
            ]
        for k in candidates:
            candidates[k] = [
                postprocess_punctuation_for_caption_metrics(
                    preprocess_sos_eos_for_scan2cap(cap)
                ) for cap in candidates[k]
            ]
        
        # Calculate scores
        score_per_caption, message, metrics = score_captions(corpus, candidates)
        metrics["total_objects_evaluated"] = len(candidates)
        
        full_message = (
            f"Dense Captioning Evaluation (Per Object, All GT Refs):\n"
            f"  Objects evaluated: {len(candidates)}\n"
            f"{message}"
        )
        
        return full_message, metrics

    def evaluate_old(self, preds: Dict[str, str], gt_indices: Dict,
                 hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Standard evaluation for dense captioning.
        Each prediction is compared against ALL ground truth captions for that object.
        This is the recommended evaluation method.
        """
        logger.info("Evaluating dense captions per object (using all GT captions as references)...")
        
        candidates = {}
        corpus = {}
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            # Extract prediction caption
            pred_caption = pred_text
            pred_caption = self._extract_caption_from_prediction(pred_text)
            
            # Get ALL GT captions for this object
            obj_key = sample["obj_key"]
            all_gt_captions = self.object_all_captions[obj_key]
            
            candidates[scanrefer_id] = [pred_caption]
            corpus[scanrefer_id] = all_gt_captions  # Use ALL GT captions as references
        
        if not candidates:
            message = "No valid predictions found to evaluate."
            logger.warning(message)
            return message, {}

        # normalize all captions
        for k in corpus:
            corpus[k] = [postprocess_punctuation_for_caption_metrics(preprocess_sos_eos_for_scan2cap(cap)) for cap in corpus[k]]
        for k in candidates:
            candidates[k] = [postprocess_punctuation_for_caption_metrics(preprocess_sos_eos_for_scan2cap(cap)) for cap in candidates[k]]
        
        # Calculate scores
        score_per_caption, message, metrics = score_captions(corpus, candidates)
        metrics["total_objects_evaluated"] = len(candidates)
        
        
        full_message = (
            f"Dense Captioning Evaluation (Per Object, All GT Refs):\n"
            f"  Objects evaluated: {len(candidates)}\n"
            f"{message}"
        )
        
        return full_message, metrics
    
    def _save_samples(self):
        """Disable saving."""
        pass
    
    def _load_samples(self):
        """Disable loading from cache."""
        pass

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)

class Real3DQADataset(Real3DDataset):
    """
    Dataset for 3D Question Answering tasks (ScanQA, SQA3D, MV-ScanQA).
    Generates CoT traces that:
    1. List all objects in the scene with their details
    2. Identify related objects based on ground truth object_ids
    3. Provide the answer to the question
    
    For SQA3D, also processes position and orientation information.
    """
    
    def __init__(
        self,
        name: str = "scanqa",  # or "sqa3d", "scanqa-mv"
        include_position_in_cot: bool = True,  # For SQA3D, whether to include position analysis in CoT
        **kwargs
    ):
        self.qa_type = self._determine_qa_type(name)
        self.include_position_in_cot = include_position_in_cot
        
        # Override defaults
        kwargs["parallel"] = False
        
        super().__init__(name=name, **kwargs)
    
    def _determine_qa_type(self, name: str) -> str:
        """Determine QA dataset type from name."""
        name_lower = name.lower()
        if "msqa" in name_lower:
            return "msqa"
        elif "sqa3d" in name_lower:
            return "sqa3d"
        elif "scanqa-mv" in name_lower or "mv-scanqa" in name_lower:
            return "scanqa-mv"
        elif "scanqa" in name_lower:
            return "scanqa"
        else:
            logger.warning(f"Unknown QA type in name '{name}', defaulting to 'scanqa'")
            return "scanqa"
    
    def _get_annotation_file(self, split):
        """Get annotation file path based on dataset type and split."""
        DSET_PATH_SCANQA = {
            "test_w_obj": f"{SVC_PATH}/ScanQA_v1.0_test_w_obj.json",
            "test_wo_obj": f"{SVC_PATH}/ScanQA_v1.0_test_wo_obj.json",
            "train": f"{SVC_PATH}/ScanQA_v1.0_train.json",
            "val": f"{SVC_PATH}/ScanQA_v1.0_val.json",
        }
        DSET_PATH_SQA3D = {
            "test": f"{self.data_path}/SQA_test_aligned.json",
            "train": f"{self.data_path}/SQA_train_aligned.json",
            "val": f"{self.data_path}/SQA_val_aligned.json",
        }
        DSET_PATH_SCANQA_MV = {
            "train": f"{SVC_PATH}/qa/ScanQA_mv_train_filtered_cleaned.json",
            "val": f"{SVC_PATH}/qa/ScanQA_mv_val_filtered_cleaned.json",
        }
        DSET_PATH_MSQA = {
            "train": f"{self.data_path}/msqa/scannet/msqa_scannet_train_aligned.json",
            "val": f"{self.data_path}/msqa/scannet/msqa_scannet_val_aligned.json",
            "test": f"{self.data_path}/msqa/scannet/msqa_scannet_test_aligned.json",
        }

        
        if self.qa_type == "sqa3d":
            return DSET_PATH_SQA3D[split]
        elif self.qa_type == "scanqa-mv":
            return DSET_PATH_SCANQA_MV[split]
        elif self.qa_type == "msqa":
            return DSET_PATH_MSQA[split] # TODO: add loading logic for MSQA 
        else:  # scanqa
            return DSET_PATH_SCANQA[split]
    
    def _initialize_templates(self, **kwargs):
        """Initialize templates for QA task."""
        # Set instruction templates BEFORE calling super()
        if self.qa_type in ["sqa3d", "msqa"]:
            kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
                "Situation: {situation}\nMy position is at {position_str}, facing {orientation_str}.\nQuestion: {question}\n",
                "Given the situation: {situation}\nI am located at {position_str}, oriented towards {orientation_str}.\n{question}\n",
                "Currently my situation: {situation}\nLocation: {position_str}, Orientation: {orientation_str}\nAnswer this question: {question}\n",
            ])
        else:  # scanqa or scanqa-mv
            kwargs["instruction_templates"] = kwargs.get("instruction_templates", [
                "Answer this question based on the room situation: {question}\n",
                "Please answer based on the room situation: {question}\n",
            ])
        
        # Call super to add common prefixes (thinking prompt + |object_set|)
        super()._initialize_templates(**kwargs)
        
        # Response templates
        self.response_templates = [
            "Roger. The answer is: {answer}",
            "Apeiria found the answer: {answer}",
            "Based on the analysis, the answer is {answer}",
            "{answer}",
        ]
        
        # Thinking trace template
        self.thinking_trace_template = [
            "[APEIRIA THINKS]\n"
            "Apeiria will now analyze the scene.\n"
            "{position_analysis}"
            "I'll examine all {object_count} objects in the scene.\n"
            "I see {object_count} object(s) in the scene: {object_id_class_info}\n"
            "{related_objects_section}"
            "Based on these information, Apeiria determined the answer is: {answer}\n"
            "Now, Apeiria will formulate the response.\n"
            "[APEIRIA SPEAKS]\n"
        ]
        
        # Related objects templates
        self.related_objects_templates = [
            "The objects related to this question are: {related_objects}\n",
            "Relevant objects for answering: {related_objects}\n",
            "Objects mentioned or related: {related_objects}\n",
        ]
        
        # Position analysis template (for SQA3D)
        self.position_analysis_templates = [
            "I am at {position_str}, facing {orientation_str}.\n",
            "My location is {position_str}, and I am oriented {orientation_str}.\n",
            "Current position: {position_str}, current orientation: {orientation_str}.\n",
        ]
        
        # Fix templates if needed
        if self.split != "train" or self.fix_template:
            self.instruction_templates = [self.instruction_templates[0]]
            self.response_templates = [self.response_templates[0]]
            self.thinking_trace_template = [self.thinking_trace_template[0]]
            self.related_objects_templates = [self.related_objects_templates[0]]
            self.position_analysis_templates = [self.position_analysis_templates[0]]
    
    def _format_position(self, position: List[float]) -> Tuple[str, str]:
        """
        Format SQA3D position into readable strings.
        
        Args:
            position: [x, y, z, qx, qy, qz, qw] where z is usually 0
            
        Returns:
            position_str: "(x, y)" formatted string
            orientation_str: "X degrees" formatted string
        """
        xy_position = position[:2]
        xy_position = [round(x, 2) for x in xy_position]
        position_str = f"({xy_position[0]}, {xy_position[1]})"
        
        # Extract quaternion and convert to yaw angle
        # Convert to Euler angles (roll, pitch, yaw) using 'sxyz' convention
        quat = position[3:7]
        roll, pitch, yaw = quat2euler(quat, axes='sxyz')
        
        # Convert yaw to degrees
        yaw_degrees = round(math.degrees(yaw), 0)
        orientation_str = f"{yaw_degrees} degrees"
        
        return position_str, orientation_str

    @staticmethod
    def _replace_object_tag(text: str) -> str:
        """
        Replace <object_name-id-IMG> tags with "object_name (id: id)" format.
        
        Args:
            text: String containing tags like <refrigerator-85-IMG>
            
        Returns:
            Cleaned text with tags replaced
            
        Examples:
            "<refrigerator-85-IMG>" -> "refrigerator (id: 85)"
            "<tissue-box-42-IMG>" -> "tissue-box (id: 42)"
        """
        # Pattern explanation:
        # <          - literal opening bracket
        # (.+?)      - capture group 1: object name (non-greedy, any characters)
        # -          - hyphen separator
        # (\d+)      - capture group 2: object id (one or more digits)
        # -IMG>      - literal suffix
        pattern = r'<(.+?)-(\d+)-IMG>'
        replacement = r'\1 (id: \2)'
        
        return re.sub(pattern, replacement, text)

    @staticmethod
    def find_object_tags(text: str) -> List[Tuple[str, int]]:
        """
        Find all object tags in the format <object_name-id-IMG>.
        
        Args:
            text: String containing tags like <refrigerator-85-IMG>
        Returns:
            List of tuples: (object_name, object_id)
        """
        pattern = r'<(.+?)-(\d+)-IMG>'
        matches = re.findall(pattern, text)
        
        result = []
        for match in matches:
            obj_name = match[0]
            obj_id = int(match[1])
            result.append((obj_name, obj_id))
        
        return result
    
    def _generate_samples(self, load_from_cache):
        """Generate QA samples with CoT."""
        logger.info(f"Generating {self.qa_type} QA samples...")
        
        samples = []
        
        for ann_idx, anno in enumerate(self.annotations):
            scene_id = anno["scene_id"]
            
            if scene_id not in self.scene_data:
                continue
            
            # Extract basic information
            question = anno["question"]
            answers = anno.get("answers", [])
            if not answers:
                logger.warning(f"No answers found for question {anno.get('question_id')}")
                continue
            
            answer = answers[0]  # Use first answer
            object_ids = anno.get("object_ids", [])
            object_names = anno.get("object_names", [])

            # for MSQA, use "raw_thought" and object tags in question/situation for "relevant objects"
            if self.qa_type == "msqa":
                object_ids, object_names = [], []
                if "raw_thought" in anno:
                    object_tags = anno["raw_thought"].split(",") # list [armchair-79, wall-19, fireplace-18, ...]
                    
                    for tag in object_tags:
                        tag = tag.strip()
                        match = re.match(r'(.+)-(\d+)', tag)
                        if match:
                            obj_name = match.group(1)
                            obj_id = match.group(2)
                            object_ids.append(int(obj_id))
                            object_names.append(obj_name)
                
                # find mentioned object tags in question and situation <object_name-id-IMG>
                question_tags = self.find_object_tags(question)
                for obj_name, obj_id in question_tags:
                    if obj_id not in object_ids:
                        object_ids.append(obj_id)
                        object_names.append(obj_name)

                situation_tags = self.find_object_tags(anno["situation"])
                for obj_name, obj_id in situation_tags:
                    if obj_id not in object_ids:
                        object_ids.append(obj_id)
                        object_names.append(obj_name)

                # Replace object tags in question and situation
                question = self._replace_object_tag(question)
                anno["situation"] = self._replace_object_tag(anno["situation"])

            question_id = anno.get("question_id", f"{scene_id}_{ann_idx}")
            
            # Get scene objects
            instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
            all_object_ids = instance_bboxes[:, 7].astype(int)
            object_classes = instance_bboxes[:, 6].astype(int)
            locations = instance_bboxes[:, :6]
            num_objects = len(object_classes)
            
            # Prepare instruction based on QA type
            if self.qa_type in ["sqa3d", "msqa"]:
                situation = anno["situation"]

                # position = anno["position"]
                # position_str, orientation_str = self._format_position(position)
                position = anno["position_aligned"]
                orientation = anno["yaw_aligned"]

                xy_position = position[:2]
                xy_position = [round(x, 2) for x in xy_position]
                position_str = f"({xy_position[0]}, {xy_position[1]})"
                orientation_str = f"{round(orientation, 0)} degrees"

                if self.qa_type == "msqa":
                    # <refrigerator-85-IMG> -> refrigerator (id: 85)
                    question = self._replace_object_tag(question)
                    situation = self._replace_object_tag(situation)
                
                description = random.choice(self.instruction_templates).format(
                    situation=situation,
                    position_str=position_str,
                    orientation_str=orientation_str,
                    question=question
                )
            else:
                description = random.choice(self.instruction_templates).format(
                    question=question
                )
            
            # Generate CoT if enabled
            if self.add_thinking_trace:
                # Create detailed object list
                # To prevent overlong CoT, only use object id and name, no size/location
                object_id_class_info = []
                for i in range(num_objects):
                    obj_name = self.object_classes[int(object_classes[i])]
                    obj_id = all_object_ids[i]
                    obj_str = self.object_id_with_class_templates.format(
                        id=obj_id,
                        object_name=obj_name
                    )
                    object_id_class_info.append(obj_str)
                
                object_id_class_info_str = ", ".join(object_id_class_info)
                
                # Create related objects section
                if object_ids:
                    # Find related object details
                    related_obj_strs = []
                    for obj_id in sorted(set(object_ids)):
                        # Find this object in the scene
                        for i, scene_obj_id in enumerate(all_object_ids):
                            if scene_obj_id == obj_id:
                                # use detailed object string on related objects
                                obj_name = self.object_classes[int(object_classes[i])]
                                obj_detail = random.choice(self.object_detail_with_class_templates).format(
                                    id=obj_id,
                                    object_name=obj_name,
                                    x=locations[i, 0],
                                    y=locations[i, 1],
                                    z=locations[i, 2],
                                    width=locations[i, 3],
                                    height=locations[i, 4],
                                    depth=locations[i, 5]
                                )
                                related_obj_strs.append(obj_detail)
                                break
                    
                    if related_obj_strs:
                        related_objects_section = random.choice(self.related_objects_templates).format(
                            related_objects="\n".join(related_obj_strs)
                        )
                    else:
                        related_objects_section = ""
                else:
                    # TODO: For SQA3D without object_ids, maybe use some heuristic or leave empty
                    if self.qa_type in ["sqa3d", "msqa"]:
                        # TODO: No ground truth object IDs available for this SQA3D question.
                        related_objects_section = "(I need think about related objects...)\n"
                    else:
                        related_objects_section = ""
                
                # Position analysis for SQA3D/MSQA
                position_analysis = ""
                if self.qa_type in ["sqa3d", "msqa"] and self.include_position_in_cot:
                    # position = anno.get("position", [0, 0, 0, 0, 0, 0, 1])
                    # position_str, orientation_str = self._format_position(position)
                    position = anno["position_aligned"]
                    orientation = anno["yaw_aligned"]
                    xy_position = position[:2]
                    xy_position = [round(x, 2) for x in xy_position]
                    position_str = f"({xy_position[0]}, {xy_position[1]})"
                    orientation_str = f"{round(orientation, 0)} degrees"
                    
                    position_analysis = random.choice(self.position_analysis_templates).format(
                        position_str=position_str,
                        orientation_str=orientation_str
                    )
                
                # Generate thinking trace
                thinking_trace = random.choice(self.thinking_trace_template).format(
                    object_count=num_objects,
                    object_id_class_info=object_id_class_info_str,
                    related_objects_section=related_objects_section,
                    answer=answer,
                    position_analysis=position_analysis
                )
                
                # Create response
                response_body = random.choice(self.response_templates).format(answer=answer)
                expected_response = thinking_trace + response_body
            else:
                # No CoT, just answer
                expected_response = random.choice(self.response_templates).format(answer=answer)
            
            # Create sample
            sample_id = len(samples)
            scanrefer_id = f"{scene_id}|{sample_id}"
            hash_id = f"real_qa_{self.qa_type}_{scene_id}_{sample_id}"
            
            samples.append({
                "scene_id": scene_id,
                "ann_id": str(sample_id),
                "description": description,
                "raw_description": question,
                "program": "",  # No program for QA # TODO
                "question": question,
                "answer": answer,
                "answers": answers,  # Keep all answers for evaluation
                "object_ids": object_ids,
                "object_names": object_names,
                "object_id": object_ids[0] if object_ids else 0,
                "scanrefer_id": scanrefer_id,
                "hash_id": hash_id,
                "expected_response": expected_response,
                "question_id": str(question_id),
                "raw_question_id": str(question_id),
                "scan2cap_id": f"{scene_id}|{question_id}|qa",
                "data_type": self.name,
                "split": self.split,
            })
        
        logger.info(f"Generated {len(samples)} {self.qa_type} QA samples")
        return samples
    
    def evaluate(self, preds: Dict[str, str], gt_indices: Dict,
                 hash_id_index: bool = False, **kwargs) -> Tuple[str, Dict]:
        """
        Evaluate QA predictions.
        Compares predicted answers with ground truth answers.
        """
        logger.info(f"Evaluating {self.qa_type} QA predictions...")
        
        total = 0
        correct_exact = 0
        correct_relaxed = 0  # Allow partial match

        candidates, corpus = {}, {}
        
        for key, pred_text in preds.items():
            scanrefer_id = key
            if hash_id_index:
                for sample in self.samples:
                    if sample["hash_id"] == key:
                        scanrefer_id = sample["scanrefer_id"]
                        break
            
            if scanrefer_id not in self.scanrefer_id_to_idx:
                logger.warning(f"Sample '{scanrefer_id}' not found. Skipping.")
                continue
            
            sample = self.samples[self.scanrefer_id_to_idx[scanrefer_id]]
            
            # Extract answer from prediction
            pred_answer = pred_text.strip().lower()

            # remove CoT parts if present
            if re.search(r"\[APEIRIA SPEAKS\]", pred_answer, re.IGNORECASE):
                parts = re.split(r"\[APEIRIA SPEAKS\]", pred_answer, flags=re.IGNORECASE)
                pred_answer = parts[-1].strip()

            # Try to extract just the answer part after "answer is:"
            answer_patterns = [
                r"answer\s+is[:\s]+(.+?)(?:\n|$)",           # "answer is: X" or "answer is X"
                r"found\s+the\s+answer[:\s]+(.+?)(?:\n|$)",  # "found the answer: X"
            ]

            # if no match, just use the whole pred_answer (after CoT removal)
            for pattern in answer_patterns:
                match = re.search(pattern, pred_answer, re.IGNORECASE)
                if match:
                    pred_answer = match.group(1).strip()
                    break
            
            # Clean up common artifacts
            pred_answer = pred_answer.rstrip('.!?,;')
            
            # Get ground truth answers
            gt_answers = sample["answers"]
            gt_answers_lower = [ans.lower().strip() for ans in gt_answers]
            
            total += 1
            
            # Exact match
            if pred_answer in gt_answers_lower:
                correct_exact += 1
                correct_relaxed += 1
            else:
                # Relaxed match: check if any GT answer is contained in prediction or vice versa
                relaxed_match = False
                for gt_ans in gt_answers_lower:
                    gt_ans_shrink = "".join(gt_ans.split())  # remove extra spaces
                    pred_answer_shrink = "".join(pred_answer.split())
                    if gt_ans_shrink in pred_answer_shrink or pred_answer_shrink in gt_ans:
                        relaxed_match = True
                        break
                
                if relaxed_match:
                    correct_relaxed += 1

            candidates[scanrefer_id] = [pred_answer]
            corpus[scanrefer_id] = gt_answers_lower

        # Calculate exact/relaxed accuracy metrics
        exact_accuracy = correct_exact / total if total > 0 else 0
        relaxed_accuracy = correct_relaxed / total if total > 0 else 0
        
        # Calculate caption similarity scores (BLEU, ROUGE, CIDEr, METEOR)
        score_per_caption, caption_message, caption_metrics = score_captions(corpus, candidates)
        
        # Combine all metrics
        metrics = {
            # Accuracy metrics
            "exact_accuracy": exact_accuracy,
            "relaxed_accuracy": relaxed_accuracy,
            "correct_exact": correct_exact,
            "correct_relaxed": correct_relaxed,
            "total_evaluated": total,
        }
        
        # Add caption similarity metrics
        metrics.update(caption_metrics)
        
        # Combine messages
        message = (
            f"{self.qa_type.upper()} QA Evaluation ({total} questions):\n"
            f"\n"
            f"Accuracy Metrics:\n"
            f"  Exact Match Accuracy: {exact_accuracy:.4f} ({correct_exact}/{total})\n"
            f"  Relaxed Match Accuracy: {relaxed_accuracy:.4f} ({correct_relaxed}/{total})\n"
            f"\n"
            f"Caption Similarity Metrics:\n"
            f"{caption_message}"
        )
        
        return message, metrics
    
    def _save_samples(self):
        """Disable saving."""
        pass
    
    def _load_samples(self):
        """Disable loading from cache."""
        pass

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)

class Real3DPreferenceDataset(Real3DDataset):
    """
    从 Real3D 数据和外部思维链创建用于 DPO 训练的偏好数据集。
    每个样本包含一个提示（prompt）、一个“chosen”响应（正确的思维链）和
    一个“rejected”响应（从同一原始样本中随机选择的错误思维链）。
    """

    def __init__(
        self,
        external_traces_path: str = None,
        external_traces_dict: Dict[str, List[Dict]] = None,
        **kwargs
    ):
        """
        初始化偏好数据集。

        Args:
            external_traces_path (str): 包含外部思维链的 JSON 文件路径。
            external_traces_dict (Dict): 将样本键映射到思维链列表的字典。
            **kwargs: 传递给父类 Real3DDataset 的其他参数。

        预期的外部 JSON 格式:
        ```json
        {
            "scene0000_00|0": [
                {
                    "thinking_trace": "正确的思维链...",
                    "thinking_trace_parts": {...},
                    "is_correct": true
                },
                {
                    "thinking_trace": "错误的思维链...",
                    "thinking_trace_parts": {...},
                    "is_correct": false
                }
            ]
        }
        ```
        注意这里的thinking_trace包含完整的模型回复，包括conclusion/response的内容，而不是单独的思维链部分。
        """
        self.external_traces_path = external_traces_path
        self.external_traces_dict = external_traces_dict

        # 确保在配对时可以使用完整的样本列表
        kwargs["parallel"] = True # - use Real3DDataset's parallel processing func, which is always correct (its non-parallel version might not match the same function)
        # 禁用父类中的思维链生成，因为我们将使用外部的
        kwargs["add_thinking_trace"] = False
        kwargs["load_from_cache"] = False

        # 首先调用父类构造函数来加载所有基础数据（场景、注释等）
        super().__init__(**kwargs)

        # 如果提供了路径，则加载外部思维链
        if self.external_traces_path and not self.external_traces_dict:
            self.external_traces_dict = self._load_external_traces(self.external_traces_path)
            self._deduplicate_external_traces()  # 去重外部思维链

        # 重写 self.samples 以包含偏好对
        self.samples = self._generate_preference_samples()

    def _load_external_traces(self, path: str) -> Dict[str, List[Dict]]:
        """从 JSON 文件加载外部思维链。"""
        logger.info(f"正在从 {path} 加载外部思维链")
        with open(path, 'r') as f:
            traces = json.load(f)

        # TODO: add format check
        total_traces = sum(len(trace_list) for trace_list in traces.values())
        logger.info(f"为 {len(traces)} 个独立样本加载了 {total_traces} 条外部思维链")
        return traces

    def _deduplicate_external_traces(self):
        # if for one sample, two traces are provided and exactly the same,
        # we can remove one of them to avoid duplicates in the dataset
        new_external_traces_dict = {}
        for key, trace_list in self.external_traces_dict.items():
            all_traces = set()
            deduped_traces = []
            for trace in trace_list:
                thinking_trace = trace.get("thinking_trace", "")
                if thinking_trace not in all_traces and thinking_trace.strip():
                    all_traces.add(thinking_trace)
                    deduped_traces.append(trace)

            new_external_traces_dict[key] = deduped_traces

        logger.info(f"去重后外部思维链数量: {sum(len(v) for v in new_external_traces_dict.values())} 条")

        self.external_traces_dict = new_external_traces_dict

    def _generate_preference_samples(self):
        """
        通过将正确的思维链与错误的思维链配对来生成偏好样本。
        """
        if not self.external_traces_dict:
            logger.error("未提供外部思维链，无法生成偏好数据集。")
            return []

        logger.info("正在生成偏好对...")
        preference_samples = []
        
        # self.samples当前包含由父类生成的基准样本
        base_samples = self.samples

        # show samples with at least one trace correct and one trace incorrect
        logger.info(f"基准样本数量: {len(base_samples)}")
        has_correct_and_incorrect = sum(
            1 for traces in self.external_traces_dict.values()
            if any(t.get("is_correct", False) for t in traces) and
                any(not t.get("is_correct", False) for t in traces)
        )
        logger.info(f"具有正确和错误思维链的样本数量: {has_correct_and_incorrect}")
        
        for base_sample in base_samples:
            trace_key = base_sample["scanrefer_id"]
            
            if trace_key not in self.external_traces_dict:
                continue

            all_traces = self.external_traces_dict[trace_key]
            correct_traces = [t for t in all_traces if t.get("is_correct", False)]
            incorrect_traces = [t for t in all_traces if not t.get("is_correct", False)]

            # 如果没有正确或错误的思维链，则无法创建配对
            if not correct_traces or not incorrect_traces:
                continue

            # 为每个正确的思维链创建一个偏好对
            for i, chosen_trace_data in enumerate(correct_traces):
                # 从错误的思维链中随机选择一个作为 rejected
                # rejected_trace_data = random.choice(incorrect_traces)
                if i > len(incorrect_traces) - 1:
                    # 如果没有足够的错误思维链，则跳过
                    # logger.warning(f"样本 {trace_key} 的正确思维链数量超过错误思维链数量，跳过。")
                    continue

                rejected_trace_data = incorrect_traces[i % len(incorrect_traces)]  # 循环使用错误思维链

                # 从基准样本和思维链数据构建响应
                response_body = base_sample["expected_response"]
                chosen_response = chosen_trace_data["thinking_trace"] # + "\n" + response_body # default thinking_trace already contains the full response
                rejected_response = rejected_trace_data["thinking_trace"] # + "\n" + response_body

                # 构造偏好数据点
                preference_data = {
                    "prompt": base_sample["description"],
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    # 仅存储 scene_id，以便在 __getitem__ 中检索特征
                    "scene_id": base_sample["scene_id"],
                    "scanrefer_id": f"{base_sample['scanrefer_id']}_pref{len(preference_samples)}", # 创建唯一ID
                }
                preference_samples.append(preference_data)

        logger.info(f"创建了 {len(preference_samples)} 个偏好对。")
        return preference_samples


    def __getitem__(self, idx):
        """返回一个偏好对，并在此时动态加载3D特征。"""
        sample_data = self.samples[idx]
        scene_id = sample_data["scene_id"]

        # 从 scene_id 动态加载3D特征
        object_feature = self.frozen_features[scene_id][0]
        object_mask = self.frozen_features[scene_id][1]
        predicted_bbox_corners = self.frozen_features[scene_id][2]
        input_predicted_bbox = torch.tensor(self.input_predicted_bboxes[scene_id])
        object_labels = self.scene_data[scene_id]["instance_bboxes"][:, 6].astype(np.int64)

        
        # 将加载的特征添加到样本中
        sample_data.update({
            "object_feature": object_feature,
            "object_mask": object_mask,
            "predicted_bbox_corners": predicted_bbox_corners,
            "input_predicted_bbox": input_predicted_bbox,
            "object_labels": object_labels,
        })

        # TODO: load 2D feature???
        if self.image_encoder_name:
            image_ids = self._get_image_indices(sample_data)
            sample_data["image_embeds"] = self._get_image_features(scene_id, image_ids) # [N_views, L, D] or [N_views, 1, D]

        return sample_data

    def __len__(self):
        """返回偏好对的总数。"""
        return len(self.samples)

    def _save_samples(self):
        raise NotImplementedError("默认情况下未实现保存偏好数据集。")

    def _load_samples(self):
        raise NotImplementedError("默认情况下未实现加载偏好数据集。")

    # use DefaultViewSelectionMixin's view selection method
    def _get_view_annotation_file(self, n_views_in_m_views):
        return DefaultViewSelectionMixin._get_view_annotation_file(self, n_views_in_m_views)
    
    def _get_image_indices(self, data):
        return DefaultViewSelectionMixin._get_image_indices(self, data)

class Real3DGroundingWithCaptionCoTDataset(Real3DDataset):
    """
    Grounding task with Observation-based Chain-of-Thought.
    Features:
    1. Context Construction: Aggregates dense captions from MULTIPLE sources (ScanRefer, Sr3D, etc.).
    2. Multi-Target Support: Handles instructions targeting multiple objects (e.g., Multi3DRef).
    3. Workflow: List Observations -> Reason -> Conclude -> Standard Response.
    """

    # Mapping dataset names to filenames (relative to data_path)
    # Supports automatic resolution based on split
    DATASET_FILE_MAP = {
        "sr3d": {
            "train": "sr3d_with_programs_train_enriched.json",
            "val": "sr3d_with_programs_val_enriched.json"
        },
        "nr3d": {
            "train": "nr3d_train_with_program.json",
            "val": "nr3d_val_with_program.json"
        },
        "scanrefer": {
            "train": "ScanRefer_filtered_train_with_program.json",
            "val": "ScanRefer_filtered_val_with_program.json"
        },
        "multi3drefer": {
            "train": "multi3drefer/multi3drefer_train.json",
            "val": "multi3drefer/multi3drefer_val.json"
        }
    }

    def __init__(
        self,
        name: str = "sr3d_caption_cot",
        caption_sources: List[str] = None, # List of names (e.g., ["scanrefer", "sr3d"]) or paths
        max_captions_per_object_in_cot: int = 10, # Limit to avoid context explosion
        max_caption_cot_len: int = 5000, # Max length for CoT context
        **kwargs
    ):
        # Default to itself
        self.caption_sources = caption_sources or []
        _this_densecap_name = name.replace("_caption_cot", "")
        if _this_densecap_name not in self.caption_sources:
            self.caption_sources.append(_this_densecap_name)
        self.caption_sources = list(set(self.caption_sources))  # Deduplicate

        self.max_captions_per_object_in_cot = max_captions_per_object_in_cot
        self.max_caption_cot_len = max_caption_cot_len
        
        # We need data_path and split immediately to load captions before parent init generates samples
        self.data_path = kwargs.get("data_path", DATA_PATH)
        self.split = kwargs.get("split", "train")
        
        # 1. Load and merge captions from all sources
        self.caption_lookup = self._load_all_caption_sources()

        # 2. Define CoT Template (Updated for multi-object conclusion)
        self.caption_cot_template = (
            "[APEIRIA THINKS]\n"
            "I need to find the object(s) described as: \"{description}\"\n"
            "First, let me examine the objects in the scene with their detailed descriptions:\n"
            "{object_list_with_captions}\n"
            "Based on the instruction \"{description}\" and the object details above, "
            "{conclusion}\n"
            "[APEIRIA SPEAKS]"
        )

        # 3. Enforce CoT settings
        kwargs["add_thinking_trace"] = True 
        kwargs["parallel"] = False # Logic is complex, prefer serial safety for generation
        
        super().__init__(name=name, **kwargs)

    def _resolve_caption_file(self, source_name):
        """Resolve dataset name to file path using the map, or return as is if it's a path."""
        # Clean up name (case insensitive match could be added here)
        key = source_name.lower()
        
        if key in self.DATASET_FILE_MAP:
            # Check if specific split exists, otherwise fallback or error
            # For CoT context, we usually want as much data as possible, so maybe use train for everything?
            # But let's stick to the current split to avoid leakage if doing strict eval.
            filename = self.DATASET_FILE_MAP[key].get(self.split)
            if not filename:
                logger.warning(f"No file defined for dataset '{key}' split '{self.split}'. skipping.")
                return None
            return os.path.join(self.data_path, filename)
        else:
            # Assume it's a direct path
            return source_name

    def _load_all_caption_sources(self):
        """
        Load multiple caption files and merge them.
        Returns: scene_id -> object_id -> list of unique captions
        """
        merged_lookup = defaultdict(lambda: defaultdict(list))
        
        for source in self.caption_sources:
            file_path = self._resolve_caption_file(source)
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"Caption source file not found: {file_path}")
                continue
                
            logger.info(f"Loading dense captions from {file_path}...")
            try:
                with open(file_path, 'r') as f:
                    raw_data = json.load(f)
                
                count = 0
                for entry in raw_data:
                    sid = entry.get("scene_id")
                    # Handle Multi3DRef (object_ids list) and Standard (object_id int)
                    target_ids = []
                    if "object_ids" in entry and isinstance(entry["object_ids"], list):
                        target_ids = [str(oid) for oid in entry["object_ids"]]
                    elif "object_id" in entry:
                        target_ids = [str(entry["object_id"])]
                    
                    desc = entry.get("description") or entry.get("question") # Compatible with QA datasets too
                    if not desc and "sentences" in entry: # Multi3DRef sentences
                        desc = " ".join(entry["sentences"])

                    if sid and target_ids and desc:
                        # Distribute caption to all relevant objects
                        for oid in target_ids:
                            # Avoid exact duplicates for the same object
                            if desc not in merged_lookup[sid][oid]:
                                merged_lookup[sid][oid].append(desc)
                        count += 1
                logger.info(f"Loaded {count} entries from {source}.")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return merged_lookup

    def _generate_samples(self, load_from_cache):
        """
        Generate grounding samples with Multi-Source Caption-CoT.
        """
        logger.info(f"Generating Multi-Source Caption-CoT samples from {self.annotation_file}...")
        samples = []

        for idx, anno in enumerate(tqdm(self.annotations)):
            scene_id = anno["scene_id"]
            if scene_id not in self.scene_data:
                continue

            # --- 1. Instruction & Targets ---
            description = anno["description"]
            
            # Robustly retrieve target IDs (handle both single and multi-target datasets)
            target_object_ids = []
            if "object_ids" in anno and isinstance(anno["object_ids"], list):
                target_object_ids = [int(oid) for oid in anno["object_ids"]]
            elif "object_id" in anno:
                target_object_ids = [int(anno["object_id"])]
            
            if not target_object_ids:
                continue

            # Get scene objects (normalized)
            scene_objects = self.scene_data[scene_id]["objects"]
            
            # Verify targets exist in scene and get their names
            valid_targets = []
            target_names = []
            for tid in target_object_ids:
                obj_data = next((obj for obj in scene_objects if str(obj["id"]) == str(tid)), None)
                if obj_data:
                    valid_targets.append(obj_data)
                    target_names.append(obj_data["name"])
            
            if not valid_targets:
                logger.warning(f"No valid target objects found in scene {scene_id} for annotation {idx}. Skipping. Could be buggy.")
                continue
                
            # Use the first target for single-target compatibility fields, but keep list for logic
            primary_target_id = valid_targets[0]["id"]
            primary_target_name = valid_targets[0]["name"]

            # --- 2. Build CoT Content (Listing Phase) ---
            object_lines = []
            
            # Iterate over ALL objects in the scene to provide full context
            for obj in scene_objects:
                obj_id = str(obj["id"])
                obj_name = obj["name"]
                
                # Format Location
                loc = obj.get("location", [0,0,0])
                loc_str = f"({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f})"
                
                # Retrieve Captions from our merged lookup
                captions = self.caption_lookup.get(scene_id, {}).get(obj_id, [])

                # shuffle to introduce randomness in selection
                random.shuffle(captions)
                
                if not captions:
                    caption_str = f"A {obj_name}."
                else:
                    # Smart selection could happen here, currently just truncation
                    selected = captions[:self.max_captions_per_object_in_cot]
                    caption_str = "; ".join(selected)
                
                line = f"Object {obj_id} ({obj_name}, {loc_str}): {caption_str}"
                object_lines.append(line)

            object_list_str = "\n".join(object_lines)

            # --- 3. Build CoT Conclusion ---
            # Handle multiple targets in the reasoning text
            target_info_strs = [f"Object {obj['id']} ({obj['name']})" for obj in valid_targets]
            if len(valid_targets) == 1:
                conclusion_str = f"the target object is identified as {target_info_strs[0]}."
            else:
                conclusion_str = f"the target objects are identified as: {', '.join(target_info_strs)}."

            thinking_trace = self.caption_cot_template.format(
                description=description,
                object_list_with_captions=object_list_str,
                conclusion=conclusion_str
            )

            # --- 4. Build Standard Response ---
            # Generate detail lines for ALL targets
            response_detail_lines = []
            for obj in valid_targets:
                loc = obj.get("location", [0,0,0])
                size = obj.get("size", [0,0,0])
                line = random.choice(self.object_detail_templates).format(
                    id=obj["id"],
                    x=loc[0], y=loc[1], z=loc[2],
                    width=size[0], height=size[1], depth=size[2]
                )
                response_detail_lines.append(line)
            
            response_body = random.choice(self.response_templates).format(
                count=len(valid_targets),
                object_details="\n".join(response_detail_lines)
            )

            expected_response = thinking_trace + "\n" + response_body

            # filter by max length
            if self.max_caption_cot_len > 0 and _fast_word_count(expected_response) > self.max_caption_cot_len:
                logger.warning(f"Skipping sample {scene_id}|{idx} due to exceeding max CoT length ({_fast_word_count(expected_response)} > {self.max_caption_cot_len})")
                continue

            # --- 5. Store Sample ---
            scanrefer_id = f"{scene_id}|{idx}"
            prompt = random.choice(self.instruction_templates).format(description=description)

            samples.append({
                "scene_id": scene_id,
                "ann_id": str(idx),
                "description": prompt,
                "raw_description": description,
                "program": anno.get("program", ""),
                "object_id": primary_target_id, # Keep for compatibility
                "object_ids": [obj["id"] for obj in valid_targets], # Full list
                "scanrefer_id": scanrefer_id,
                "hash_id": f"real_grounding_cot_{scene_id}_{idx}",
                "expected_response": expected_response,
                "question_id": f"{scene_id}_{idx}",
                "raw_question_id": f"{scene_id}_{idx}",
                "scan2cap_id": f"{scene_id}|{primary_target_id}|{primary_target_name}", # Primary ID for lookup
                "data_type": self.name,
                "split": self.split,
                "thinking_trace_parts": {
                    "all": expected_response,
                    "header": f"[APEIRIA THINKS]\nI need to find the object(s) described as: \"{description}\"",
                    "execution": object_list_str,
                    "conclusion": conclusion_str
                }
            })

        # simple stat of CoT lengths
        # use _fast_len to avoid counting unicode chars
        cot_lengths = [_fast_word_count(sample["expected_response"]) for sample in samples]
        # mean, max, median
        mean_len = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
        max_len = max(cot_lengths) if cot_lengths else 0
        median_len = sorted(cot_lengths)[len(cot_lengths)//2] if cot_lengths else 0
        logger.info(f"[{self.name} - {self.split}] CoT Lengths - Mean: {mean_len:.1f}, Max: {max_len}, Median: {median_len}")

        logger.info(f"Generated {len(samples)} Multi-Source Caption-CoT samples.")
        return samples

    
Synthetic3DDatasetType = Union[
    Synthetic3DDataset, Synthetic3DObjectInfoDataset, Synthetic3DRelationalDataset, 
    Real3DDataset, Real3DObjectInfoDataset, Real3DFilterDataset, Real3DDatasetWithExternalTrace,
    Real3DPreferenceDataset, Real3DDatasetWithAttributesNew, Real3DDatasetWithAttributes,
]

# Helper function to convert single-trace format to multi-trace format
def convert_to_multi_trace_format(single_trace_dict: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """
    Convert single-trace format to multi-trace format.
    Each entry becomes a list with one trace.
    """
    multi_trace_dict = {}
    for key, trace_data in single_trace_dict.items():
        if trace_data.get("thinking_trace"):  # Only add if trace exists
            multi_trace_dict[key] = [trace_data]
        else:
            multi_trace_dict[key] = []  # Empty list for removal
    
    return multi_trace_dict


# Helper function to merge multiple trace files
def merge_trace_files(trace_files: List[str], output_path: str):
    """
    Merge multiple trace files into one multi-trace format file.
    Useful for combining traces from different sources/models.
    """
    merged_traces = defaultdict(list)
    
    for trace_file in trace_files:
        with open(trace_file, 'r') as f:
            traces = json.load(f)
        
        # Handle both single-trace and multi-trace formats
        for key, trace_data in traces.items():
            if isinstance(trace_data, list):
                # Already multi-trace format
                merged_traces[key].extend(trace_data)
            else:
                # Single-trace format
                if trace_data.get("thinking_trace"):
                    merged_traces[key].append(trace_data)
    
    # Convert defaultdict to regular dict
    merged_traces = dict(merged_traces)
    
    # Save merged traces
    with open(output_path, 'w') as f:
        json.dump(merged_traces, f, indent=2)
    
    total_traces = sum(len(trace_list) for trace_list in merged_traces.values())
    logger.info(f"Merged {len(trace_files)} files into {output_path}")
    logger.info(f"Total: {total_traces} traces for {len(merged_traces)} unique samples")
    
    return merged_traces


# Helper function to convert internal traces to external format
def export_thinking_traces_from_dataset(dataset: Real3DDataset, output_path: str):
    """
    Export thinking traces from an existing Real3DDataset to external format.
    This can be used to generate initial external traces.
    """
    external_traces = {}
    
    for sample in dataset.samples:
        trace_key = sample["scanrefer_id"]
        
        # Extract thinking trace from expected_response
        expected_response = sample.get("expected_response", "")
        thinking_parts = sample.get("thinking_trace_parts", {})
        
        # Find thinking trace in expected response
        thinking_trace = ""
        if "[APEIRIA THINKS]" in expected_response and "[APEIRIA SPEAKS]" in expected_response:
            start_idx = expected_response.find("[APEIRIA THINKS]")
            end_idx = expected_response.find("[APEIRIA SPEAKS]") + len("[APEIRIA SPEAKS]")
            thinking_trace = expected_response[start_idx:end_idx]
        
        external_traces[trace_key] = {
            "thinking_trace": thinking_trace,
            "thinking_trace_parts": thinking_parts,
            "metadata": {
                "scene_id": sample["scene_id"],
                "description": sample["description"],
                "program": sample.get("program", ""),
                "object_id": sample.get("object_id", None)
            }
        }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(external_traces, f, indent=2)
    
    logger.info(f"Exported {len(external_traces)} thinking traces to {output_path}")
    return external_traces



if __name__ == "__main__":
    # simple configure logging
    logging.basicConfig(level=logging.INFO)

    # Test Real3DDataset
    train_dataset = Real3DDataset(split="train", name="sr3d", ratio=1, add_thinking_trace=True, parallel=True)
    dataset = Real3DDataset(split="val", name="sr3d", ratio=1, add_thinking_trace=True, parallel=True)
    logger.info(f"Loaded {len(dataset)} validation samples and {len(train_dataset)} training samples")

    input()
    
    # Get a sample
    sample = dataset[0]
    logger.info(sample)

    
    # Evaluate predictions
    preds = {
        "scene_id|0": "Object 0: At (0.0, 0.0, 0.0), size: 1.0 x 1.0 x 1.0",
        "scene_id|1": "ID 1: Position (0.0, 0.0, 0.0), size 1.0 x 1.0 x 1.0",
        "scene_id|2": "2: Coordinates (0.0, 0.0, 0.0), dimensions 1.0 x 1.0 x 1.0",
        "scene_id|3": "Object 3: (0.0, 0.0, 0.0), 1.0 x 1.0 x 1.0"
    }
    gt_indices = {
        "scene_id|0": [0],
        "scene_id|1": [1],
        "scene_id|2": [2],
        "scene_id|3": [3]
    }
    message, metrics = dataset.evaluate(preds, gt_indices)
    logger.info(message)
    logger.info(metrics)
    
    # Check if the dataset can be loaded in a DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in loader:
        logger.info(batch)
        break