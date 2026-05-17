import numpy as np
import json
import argparse
import os

class CustomEncoder(json.JSONEncoder):
    def __init__(self, precision=2, *args, **kwargs):
        # 提取 precision 参数，默认为 2
        self.precision = precision
        # 预定义格式化字符串，例如 "{:.2f}"
        self.fmt = f"{{:.{self.precision}f}}"
        super().__init__(*args, **kwargs)
        self.indent_level = 0
        self.indent_str = kwargs.get('indent', 0) * ' '

    def encode(self, obj):
        if isinstance(obj, dict):
            self.indent_level += 1
            output = '{\n' + ',\n'.join(f'{self.indent_str * self.indent_level}"{k}": {self.encode(v)}' for k, v in obj.items()) + '\n' + self.indent_str * (self.indent_level - 1) + '}'
            self.indent_level -= 1
            return output
        elif isinstance(obj, list):
            # 检查是否为纯浮点数列表（用于坐标或尺寸）
            if len(obj) in [2,3] and all(isinstance(x, float) for x in obj):
                # 使用动态精度格式化
                return f'[{", ".join(self.fmt.format(x) for x in obj)}]'
            self.indent_level += 1
            output = '[\n' + ',\n'.join(f'{self.indent_str * self.indent_level}{self.encode(item)}' for item in obj) + '\n' + self.indent_str * (self.indent_level - 1) + ']'
            self.indent_level -= 1
            return output
        elif isinstance(obj, float):
            # 单个浮点数也使用动态精度
            return self.fmt.format(obj)
        return super().encode(obj)
    

def get_attributes(class_name):
    default_attributes = ["placeholder"]
    attribute_map = {
        "apple": ["red"],
        "desk": ["brown", "wooden"],
        "tv": ["black", "off"],
        "cabinet": ["light brown", "wooden"],
        "cup": ["glass", "empty"],
        "guitar": ["brown", "wooden"]
    }
    return attribute_map.get(class_name, default_attributes)


def main(input_file, output_file, caption_file=None, precision=2):
    # 加载物体描述数据
    if caption_file:
        with open(caption_file, 'r') as f:
            object_captions = json.load(f)

    # Load the bounding box data
    bbox_data = np.load(input_file)
    
    if 'scans_fixed' in input_file:
        old_file = input_file.replace("scannet/scans_fixed", "data/scannet_data")
        bbox_data_old = np.load(old_file)
    else:
        old_file = input_file.replace("scannet/scans", "data/scannet_data")

    bbox_data_old = np.load(old_file)

    # /home/mwt/hdd/apeiria/scannet/scans/scene0011_00_aligned_bbox.npy -> /home/mwt/hdd/apeiria/scannet/scans/scene0011_00_categories.json
    # Load raw category
    raw_category = input_file.replace("_aligned_bbox.npy", "_categories.json")
    raw_category_data = json.load(open(raw_category))

    type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}    
    class2name = {type2class[t]:t for t in type2class}
    # nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
    nyu40ids = np.array([0, 1, 2, 22, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)

    def _get_nyu40id2name():
        lines = [line.rstrip() for line in open(os.path.join('data', 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2name = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in nyu40ids:
                nyu40ids2name[nyu40_id] = nyu40_name

        return nyu40ids2name

    def _get_nyu40id2class():
        lines = [line.rstrip() for line in open(os.path.join('data', 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in nyu40ids:
                # nyu40ids2class[nyu40_id] = type2class[nyu40_name]
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = type2class[nyu40_name]

        return nyu40ids2class
    
    nyu40id2class = _get_nyu40id2class()
    nyu40id2name = _get_nyu40id2name()
    # print(nyu40id2name)

    # Prepare the scene data
    scene_data = {"objects": [], "corners": []}

    # load box data
    for i, bbox in enumerate(bbox_data):
        center_x, center_y, center_z = bbox[:3]
        size_x, size_y, size_z = bbox[3:6]
        # class_idx = int(bbox[-2])
        nyu40id = int(bbox[-2])
        # print(nyu40id)
        # class_idx = nyu40id2class[nyu40id]
        

        # input file: scene0011_00_aligned_bbox.npy
        scene_id = os.path.basename(input_file).split("_aligned_bbox.npy")[0]
        
        if always := True or class_idx in class2name:
            # object_name = class2name[class_idx]
            object_name = raw_category_data[i]

            if nyu40id not in nyu40id2name:
                print(f"nyu40id {nyu40id} not in nyu40ids")
                print(f"object name: {raw_category_data[i]}")
            # nyu40_name = class2name[class_idx]
            # nyu40_name = nyu40id2name[nyu40id]
            nyu40_name = nyu40id2name.get(nyu40id, "unannotated")
            
            object_data = {
                "name": object_name,
                "nyu40_name": nyu40_name,
                "id": int(bbox[-1]),
                "location": [float(center_x), float(center_y), float(center_z)],
                "size": [float(size_x), float(size_y), float(size_z)],
            }
            if caption_file and scene_id in object_captions and str(i) in object_captions[scene_id]:
                # search object index in old bbox data
                old_bbox_index = -1
                for j, bbox_old in enumerate(bbox_data_old):
                    if int(bbox_old[-1]) == int(bbox[-1]):
                        old_bbox_index = j

                # some objects (e.g. wall, floor, ceiling) are not in the old bbox data
                if old_bbox_index != -1:
                    captions = object_captions[scene_id][str(old_bbox_index)]
                    if len(captions) > 0:
                        # take only top 1
                        # captions = captions[:1]
                        # object_data["captions"] = [{"description": c[0], "score": c[1]} for c in captions]
                        # object_data["captions"] = [c[0] for c in captions]
                        object_data["caption"] = captions[0][0]
            
            scene_data["objects"].append(object_data)

    # load room corners data
    # /home/mwt/hdd/apeiria/data/scannet_data/scene0001_00_aligned_vert_corners.json
    corners = json.load(open(os.path.join(os.path.dirname(old_file), scene_id + "_aligned_vert_corners.json")))
    
    # corners are list of list of N 2D points, each list of N 2D points stands for a room's all corners
    scene_data["corners"] = corners

    # Write to JSON file
    with open(output_file, 'w') as f:
        # json.dump(scene_data, f, cls=CustomEncoder)
        json_string = CustomEncoder(indent=4, precision=precision).encode(scene_data)
        f.write(json_string)

    # print(f"JSON file has been created: {output_file} with {len(scene_data['objects'])} objects.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert bounding box data to JSON scene representation.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input .npy file containing bounding box data")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output JSON file name")
    parser.add_argument('-c', '--captions', type=str, help="caption JSON file")
    parser.add_argument('-p', '--precision', type=int, default=2, help="Decimal precision for float values (default: 2)")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.captions, args.precision)