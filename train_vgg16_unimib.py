import random
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# =========================
# CONFIG
# =========================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
NUM_WORKERS = 0
VAL_RATIO = 0.15
MIN_SAMPLES_PER_CLASS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
ORIGINAL_DIR = DATASET_DIR / "original"

FINAL_FOOD_LIST_PATH = DATASET_DIR / "final_food_list.mat"
TRAIN_SET_PATH = DATASET_DIR / "TrainingSet.mat"
TEST_SET_PATH = DATASET_DIR / "TestSet.mat"
ANNOTATIONS_EXPORT_PATH = DATASET_DIR / "annotations_export.mat"


# =========================
# UTILS
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_names():
    data = scipy.io.loadmat(FINAL_FOOD_LIST_PATH)
    food_list = data["final_food_list"]
    return [str(item[0][0]).strip() for item in food_list]


def load_split(mat_path, key_name):
    data = scipy.io.loadmat(mat_path)
    arr = data[key_name]
    return [str(item[0][0]).strip() for item in arr]


def load_annotations_export():
    data = scipy.io.loadmat(
        ANNOTATIONS_EXPORT_PATH,
        squeeze_me=True,
        struct_as_record=False
    )

    ann_keys = data["ann_keys"]
    ann_values = data["ann_values"]

    if np.isscalar(ann_keys):
        ann_keys = [ann_keys]
        ann_values = [ann_values]

    annotations = {}
    for key, value in zip(ann_keys, ann_values):
        annotations[str(key).strip()] = value
    return annotations


def build_image_index():
    image_index = {}
    for p in ORIGINAL_DIR.iterdir():
        if p.is_file():
            image_index[p.stem.strip()] = p
    return image_index


def read_image_unicode(path: Path):
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def annotation_rows_to_list(entry):
    if isinstance(entry, np.ndarray):
        if entry.ndim == 2:
            return [[entry[r, c] for c in range(entry.shape[1])] for r in range(entry.shape[0])]
        if entry.ndim == 1:
            return [list(entry)]
        return [[entry]]
    if isinstance(entry, list):
        return [entry]
    return [[entry]]


def extract_polygon_points(candidate):
    try:
        arr = np.array(candidate, dtype=float).flatten()
    except Exception:
        return None

    if len(arr) < 6:
        return None
    if len(arr) % 2 != 0:
        arr = arr[:-1]

    try:
        pts = arr.reshape(-1, 2)
    except Exception:
        return None
    return pts


def parse_instances_from_rows(rows):
    instances = []
    for row in rows:
        if len(row) < 5:
            continue

        try:
            class_name = str(row[1]).strip()
        except Exception:
            continue

        polygon = None

        try:
            if str(row[3]).strip() == "Polygon:":
                polygon = extract_polygon_points(row[4])
        except Exception:
            polygon = None

        if polygon is None:
            for j, item in enumerate(row):
                try:
                    txt = str(item).strip()
                except Exception:
                    txt = ""

                if txt == "Polygon:":
                    for k in range(j + 1, len(row)):
                        polygon = extract_polygon_points(row[k])
                        if polygon is not None:
                            break
                    break

        if class_name and polygon is not None:
            instances.append((class_name, polygon))

    return instances


def crop_from_polygon(image, polygon_pts, pad=5):
    h, w = image.shape[:2]

    x_min = max(int(np.floor(np.min(polygon_pts[:, 0]))) - pad, 0)
    y_min = max(int(np.floor(np.min(polygon_pts[:, 1]))) - pad, 0)
    x_max = min(int(np.ceil(np.max(polygon_pts[:, 0]))) + pad, w - 1)
    y_max = min(int(np.ceil(np.max(polygon_pts[:, 1]))) + pad, h - 1)

    if x_max <= x_min or y_max <= y_min:
        return None

    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None

    return crop


# =========================
# DATASET BUILD
# =========================
def build_samples():
    original_class_names = load_class_names()
    valid_classes = set(original_class_names)
    train_split = set(load_split(TRAIN_SET_PATH, "TrainingSet"))
    test_split = set(load_split(TEST_SET_PATH, "TestSet"))
    annotations = load_annotations_export()
    image_index = build_image_index()

    raw_train_samples = []
    raw_test_samples = []

    for image_stem, ann_entry in annotations.items():
        image_stem = str(image_stem).strip()

        if image_stem in train_split:
            split_name = "train"
        elif image_stem in test_split:
            split_name = "test"
        else:
            continue

        img_path = image_index.get(image_stem)
        if img_path is None:
            continue

        rows = annotation_rows_to_list(ann_entry)
        instances = parse_instances_from_rows(rows)

        for class_name, polygon in instances:
            if class_name not in valid_classes:
                continue

            sample = {
                "image_path": str(img_path),
                "class_name": class_name,
                "polygon": polygon.astype(np.float32),
            }

            if split_name == "train":
                raw_train_samples.append(sample)
            else:
                raw_test_samples.append(sample)

    train_counts = Counter(s["class_name"] for s in raw_train_samples)
    kept_classes = {cls for cls, cnt in train_counts.items() if cnt >= MIN_SAMPLES_PER_CLASS}

    train_samples = [s for s in raw_train_samples if s["class_name"] in kept_classes]
    test_samples = [s for s in raw_test_samples if s["class_name"] in kept_classes]

    class_names = sorted(list(kept_classes))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for s in train_samples:
        s["label"] = class_to_idx[s["class_name"]]

    for s in test_samples:
        s["label"] = class_to_idx[s["class_name"]]

    return class_names, train_samples, test_samples


def split_train_val(train_samples):
    random.shuffle(train_samples)
    split = int(len(train_samples) * (1 - VAL_RATIO))
    return train_samples[:split], train_samples[split:]


# =========================
# DATASET CLASS
# =========================
class FoodDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = read_image_unicode(Path(sample["image_path"]))
        crop = crop_from_polygon(image, sample["polygon"], pad=5)

        if crop is None:
            crop = image

        crop = Image.fromarray(crop)

        if self.transform:
            crop = self.transform(crop)

        label = sample["label"]
        return crop, label


# =========================
# TRAIN
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_labels, all_preds


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)

    class_names, train_samples, test_samples = build_samples()
    train_samples, val_samples = split_train_val(train_samples)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = FoodDataset(train_samples, train_tf)
    val_ds = FoodDataset(val_samples, eval_tf)
    test_ds = FoodDataset(test_samples, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 0
    best_path = BASE_DIR / "best_vgg16_unimib.pth"

    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

    print("Best val_acc:", best_val)

    model.load_state_dict(torch.load(best_path))

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)

    print("\nTEST RESULT acc=", test_acc)
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    main()