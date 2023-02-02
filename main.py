from collections import defaultdict
from tensorflow.lite.tools import flatbuffer_utils as utils
from openvino.runtime import Core
import wget
import sys
import os
DIRECTORY = "models"

def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def read_from_log_file(file_name: str):
    file = open(file_name)
    lines = file.readlines()
    models = set()
    for line in lines:
        if '$' not in line:
            continue
        line = line.strip()
        models.add((line.split("$")[0], line.split("$")[1]))
    print(len(models), "models collected from", file_name)
    return models

TT = utils.schema_fb.TensorType
tensor_types = {getattr(TT, name): name for name in dir(TT) if not name.startswith("_")}

def get_original_types(path: str):
    result = defaultdict(int)
    try:
        model = utils.read_model(path)
        buffers = model.buffers
        subgraphs = model.subgraphs
        for subgraph in subgraphs:
            tensors = subgraph.tensors
            for i, tensor in enumerate(tensors):
                buffer = buffers[tensor.buffer].data
                if buffer is None:
                    continue
                quantization = tensor.quantization
                type_name = tensor_types[tensor.type]
                if quantization is not None and quantization.zeroPoint is not None:
                    type_name += '_quantized'
                result[type_name] += 1
        result = dict(result)
    except Exception as e:
        result = str(e)
    return result


def per_model_test(model_path: str, core: Core):
    status = None
    original_types = None
    ov_types = None
    num_transposes = None
    try:
        original_types = get_original_types(model_path)
        model = core.read_model(model_path)
        model.validate_nodes_and_infer_types()
        num_transposes = 0
        ov_types = defaultdict(int)
        for op in model.get_ops():
            if op.type_info.name == "Transpose":
                num_transposes += 1
            if op.type_info.name == "Constant":
                ov_types[str(op.output(0).element_type)] += 1
        status = "OK"
        ov_types = dict(ov_types)
    except Exception as e:
        status = " ".join(str(e).strip().split('\n'))
    return status, original_types, ov_types, num_transposes


if __name__ == "__main__":
    models = []
    links_dir = "model_link_collections"
    for file_name in os.listdir(links_dir):
        m = read_from_log_file(os.path.join(links_dir, file_name))
        models.extend(m)
    print(len(models), "models collected from all files")
    if not os.path.isdir(DIRECTORY):
        os.makedirs(DIRECTORY)
    results = dict()
    core = Core()

    pretty_head = ["Name", "Link", "Status", "#Transposes", set(), set()]
    for name, link in models:
        print(name, link)
        model_path = wget.download(link, DIRECTORY, bar=bar_progress)
        results[(name, link)] = per_model_test(os.path.normpath(model_path), core)
        curr_result = results[(name, link)]

        # if curr_result[0] == "OK":
        #     os.remove(model_path)
        os.remove(model_path)
        print()
        print(curr_result)
        if curr_result[1] is not None: #  original types
            pretty_head[4].update(set(curr_result[1].keys()))
        if curr_result[2] is not None: # ov types
            pretty_head[5].update(set(curr_result[2].keys()))

    full_picture = list()
    with open("tflite_read_test.dsv", "w+") as f:
        for (name, link), values in results.items():
            print(name, link, values)
            f.write("$".join([name, link, *map(str, values)]) + "\n")
            line = [name, link, values[0], values[-1]]
            # original types
            for orig_type in pretty_head[4]:
                line.append(values[1].get(orig_type, None) if values[1] is not None else None)
            # ov types
            for ov_type in pretty_head[5]:
                line.append(values[2].get(ov_type, None) if values[2] is not None else None)
            full_picture.append(line)
    print("Done")
    with open("tflite_read_test_report.dsv", "w+") as f:
        pretty_head[4] = "$".join(pretty_head[4])
        pretty_head[5] = "$".join(pretty_head[5])
        f.write("$".join(pretty_head) + "\n")
        for line in full_picture:
            f.write("$".join(map(str, line)) + "\n")
