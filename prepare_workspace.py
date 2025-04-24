import os
import shutil
import random
import sys

BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MedSight_Project"))
BASE_WORKSPACE_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "MediScan", "workspace"))

def prepare_classification_workspace(project_id, source_folder, train_ratio=100, val_ratio=15, test_ratio=15):

    source_folder = os.path.join(BASE_PROJECT_DIR, project_id, "annotated_images")
    classification_folder = os.path.join(BASE_WORKSPACE_DIR, project_id, "classification")
    
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Project '{project_id}' not found at {source_folder}")
    
    if os.path.exists(classification_folder):
        shutil.rmtree(classification_folder)

    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(classification_folder, split), exist_ok=True)

    image_label_pairs = []
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(folder_path, file_name)
                    label_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + ".txt")
                    if os.path.exists(label_path):
                        image_label_pairs.append((image_path, label_path))

    if not image_label_pairs:
        raise RuntimeError("No labeled images found for classification.")

    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)

    val_count = int(total * val_ratio / 100)
    test_count = int(total * test_ratio / 100)
    train_count = total - val_count - test_count  # เหลือให้ train

    train_items = image_label_pairs[:train_count]
    val_items = image_label_pairs[train_count:train_count + val_count]
    test_items = image_label_pairs[train_count + val_count:]

    splits = {"train": train_items, "valid": val_items, "test": test_items}
    split_counts = {key: 0 for key in splits}

    for split, items in splits.items():
        for image_path, label_path in items:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                class_name = f"class_{class_id}"
                class_dir = os.path.join(classification_folder, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
                shutil.copy(image_path, os.path.join(class_dir, os.path.basename(image_path)))
                split_counts[split] += 1

    print(f"Classification Dataset prepared at: {classification_folder}")
    print(f"  Total images: {total}")
    print(f"  Train: {split_counts['train']} | Val: {split_counts['valid']} | Test: {split_counts['test']}")
    
def prepare_segment_detect_dataset(project_id, source_folder, data_yaml_source, base_workspace="./workspace", val_ratio=15, test_ratio=15):
    project_path = os.path.join(base_workspace, project_id)
    folders = {
        "train_images": os.path.join(project_path, "train/images"),
        "train_labels": os.path.join(project_path, "train/labels"),
        "val_images": os.path.join(project_path, "valid/images"),
        "val_labels": os.path.join(project_path, "valid/labels"),
        "test_images": os.path.join(project_path, "test/images"),
        "test_labels": os.path.join(project_path, "test/labels"),
    }

    # ✅ ลบโฟลเดอร์เดิม
    for key in folders:
        dir_path = os.path.dirname(folders[key])
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # ✅ สร้างโฟลเดอร์ใหม่
    for path in folders.values():
        os.makedirs(path, exist_ok=True)

    # ✅ รวบรวมภาพและ label ที่ match กัน
    image_label_pairs = []
    seen_images = set()
    for class_folder in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, class_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(folder_path, file)
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(label_path) and file not in seen_images:
                        image_label_pairs.append((img_path, label_path))
                        seen_images.add(file)

    if not image_label_pairs:
        raise RuntimeError("No labeled images found for segmentation/detection.")

    random.shuffle(image_label_pairs)
    total = len(image_label_pairs)

    val_count = int(total * val_ratio / 100)
    test_count = int(total * test_ratio / 100)
    train_count = total - val_count - test_count  # เหลือให้ train

    train_files = image_label_pairs[:train_count]
    val_files = image_label_pairs[train_count:train_count + val_count]
    test_files = image_label_pairs[train_count + val_count:]

    def copy_files(pairs, dest_img, dest_lbl):
        for img_path, lbl_path in pairs:
            shutil.copy(img_path, dest_img)
            shutil.copy(lbl_path, dest_lbl)

    copy_files(train_files, folders["train_images"], folders["train_labels"])
    copy_files(val_files, folders["val_images"], folders["val_labels"])
    copy_files(test_files, folders["test_images"], folders["test_labels"])

    # ✅ คัดลอก data.yaml
    if os.path.exists(data_yaml_source):
        shutil.copy(data_yaml_source, os.path.join(project_path, "data.yaml"))
        print(f"data.yaml copied to: {project_path}")
    else:
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_source} for project '{project_id}'")
        print("data.yaml not found!")

    print(f"Dataset for Segmentation and Detection prepared at: {project_path}")
    print(f"  Total images: {total}")
    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    
def prepare_workspace(project_id, mode):
    source_folder = os.path.join(BASE_PROJECT_DIR, project_id, "annotated_images")
    data_yaml_source = os.path.join(BASE_PROJECT_DIR, project_id, "data.yaml")

    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Project '{project_id}' not found at {source_folder}")

    if mode == "classify":
        prepare_classification_workspace(project_id, source_folder)
    else:
        prepare_segment_detect_dataset(project_id, source_folder, data_yaml_source)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_workspace.py <project_id> <mode>")
        sys.exit(1)

    project_id = sys.argv[1].strip()
    mode = sys.argv[2].strip().lower()
    print(f"Prepare workspace for project: {project_id} in mode: {mode}")
    prepare_workspace(project_id, mode)
