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
EPOCHS = 15
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
# SAMPLE BUILDING
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

    skipped_images = 0
    skipped_instances = 0

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
            skipped_images += 1
            continue

        rows = annotation_rows_to_list(ann_entry)
        instances = parse_instances_from_rows(rows)

        for class_name, polygon in instances:
            if class_name not in valid_classes:
                skipped_instances += 1
                continue

            sample = {
                "image_stem": image_stem,
                "image_path": str(img_path),
                "class_name": class_name,
                "polygon": polygon.astype(np.float32),
            }

            if split_name == "train":
                raw_train_samples.append(sample)
            else:
                raw_test_samples.append(sample)

    print(f"Raw train sample sayisi: {len(raw_train_samples)}")
    print(f"Raw test sample sayisi: {len(raw_test_samples)}")
    print(f"Skipped images: {skipped_images}")
    print(f"Skipped instances: {skipped_instances}")

    train_counts = Counter(s["class_name"] for s in raw_train_samples)
    kept_classes = {cls for cls, cnt in train_counts.items() if cnt >= MIN_SAMPLES_PER_CLASS}

    print(f"Min sample esigi: {MIN_SAMPLES_PER_CLASS}")
    print(f"Train'de tutulan class sayisi: {len(kept_classes)}")
    print(f"Train'de elenen class sayisi: {len(train_counts) - len(kept_classes)}")

    train_samples = [s for s in raw_train_samples if s["class_name"] in kept_classes]
    test_samples = [s for s in raw_test_samples if s["class_name"] in kept_classes]

    filtered_class_names = sorted(list(kept_classes))
    class_to_idx = {name: i for i, name in enumerate(filtered_class_names)}

    for s in train_samples:
        s["label"] = class_to_idx[s["class_name"]]
    for s in test_samples:
        s["label"] = class_to_idx[s["class_name"]]

    print(f"Filtered train sample sayisi: {len(train_samples)}")
    print(f"Filtered test sample sayisi: {len(test_samples)}")

    return filtered_class_names, class_to_idx, train_samples, test_samples


def split_train_val(train_samples, val_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_class = {}

    for s in train_samples:
        by_class.setdefault(s["label"], []).append(s)

    train_final = []
    val_final = []

    for _, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        val_final.extend(items[:n_val])
        train_final.extend(items[n_val:])

    rng.shuffle(train_final)
    rng.shuffle(val_final)

    return train_final, val_final


def compute_class_weights(train_samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float32)
    for s in train_samples:
        counts[s["label"]] += 1.0

    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# =========================
# DATASET
# =========================
class UNIMIBCropDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = read_image_unicode(Path(sample["image_path"]))
        if image is None:
            raise RuntimeError(f"Image okunamadi: {sample['image_path']}")

        crop = crop_from_polygon(image, sample["polygon"], pad=5)
        if crop is None:
            crop = image

        crop = Image.fromarray(crop)
        if self.transform:
            crop = self.transform(crop)

        label = sample["label"]
        return crop, label


# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

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
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return total_loss / total, correct / total, all_labels, all_preds


def main():
    set_seed(SEED)

    print("Device:", DEVICE)
    print("Original dir exists:", ORIGINAL_DIR.exists())
    print("Annotations export exists:", ANNOTATIONS_EXPORT_PATH.exists())

    class_names, class_to_idx, train_samples, test_samples = build_samples()
    train_samples, val_samples = split_train_val(train_samples, VAL_RATIO, SEED)

    print(f"Final train: {len(train_samples)}")
    print(f"Final val:   {len(val_samples)}")
    print(f"Final test:  {len(test_samples)}")
    print(f"Num classes: {len(class_names)}")

    class_weights = compute_class_weights(train_samples, len(class_names)).to(DEVICE)
    print("Class weights hazirlandi.")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = UNIMIBCropDataset(train_samples, transform=train_tf)
    val_ds = UNIMIBCropDataset(val_samples, transform=eval_tf)
    test_ds = UNIMIBCropDataset(test_samples, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_path = BASE_DIR / "best_resnet50_unimib_improved.pth"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": best_val_acc,
            }, best_path)

    print(f"\nEn iyi val_acc: {best_val_acc:.4f}")
    print(f"Model kaydedildi: {best_path}")

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    print(f"\nTEST SONUCU | loss={test_loss:.4f} acc={test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix shape:", cm.shape)


if __name__ == "__main__":
    main()