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

def per_model_test(model: str, core: Core):
    try:
        model = core.read_model(model)
        model.validate_nodes_and_infer_types()
        return "OK"
    except Exception as e:
        return " ".join(str(e).strip().split('\n'))

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
    for name, link in models:
        print(name, link)
        model_path = wget.download(link, DIRECTORY, bar=bar_progress)
        results[(name, link)] = per_model_test(os.path.normpath(model_path), core)
        if results[(name, link)] == "OK":
            os.remove(model_path)
        print(results[(name, link)])

    with open("tflite_read_test.dsv", "w+") as f:
        for (name, link), value in results.items():
            print(name, link, value)
            f.write("$".join([name, link, value]) + "\n")
    print("Done")
