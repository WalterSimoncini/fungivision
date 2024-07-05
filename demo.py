import torch
import logging
import numpy as np

from tqdm import tqdm
from typing import Tuple
from src.wrapper import FUNGIWrapper
from torch.nn.functional import normalize
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
from src.utils.metrics import mean_per_class_accuracy


logging.basicConfig(
    format="[%(asctime)s:%(levelname)s]: %(message)s",
    level=logging.INFO
)

k = 20
batch_size = 16
cache_dir = "/scratch-shared/wsimoncini/cache/flowers102"

device = torch.device("cuda")
model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")

fungi = FUNGIWrapper(
    model=model,
    target_layer="blocks.11.attn.proj",
    device=device,
    latent_dim=768,
    use_fp16=True
)

def extract_features(wrapper: FUNGIWrapper, dataset: Dataset, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gradients, targets = [], []

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        # Make sure each iteration returns a list of images and a list of targets
        collate_fn=lambda batch: zip(*batch)
    )

    for images, batch_targets in tqdm(data_loader):
        targets.append(torch.tensor(batch_targets))
        gradients.append(
            wrapper.forward(list(images)).cpu().float()
        )

    return normalize(torch.cat(gradients, dim=0), dim=-1), torch.cat(targets, dim=0)

# Create train and test datasets and loaders
train_dataset = Flowers102(root=cache_dir, split="train", download=True)
test_dataset = Flowers102(root=cache_dir, split="test", download=True)

# Compute stuff needed for the loss computation, e.g. the
# SimCLR comparison batch
fungi.setup(dataset=train_dataset)

# Extract features
logging.info("extracting the train features...")
train_features, train_targets = extract_features(wrapper=fungi, dataset=train_dataset, batch_size=batch_size)

logging.info("extracting the test features...")
test_features, test_targets = extract_features(wrapper=fungi, dataset=test_dataset, batch_size=batch_size)

# Evaluate
logging.info("fitting the knn classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(train_features, train_targets)

logging.info("testing the knn classifier...")

predictions = knn_classifier.predict(test_features)
test_targets = np.array(test_targets)

correct_predictions = (predictions == test_targets).sum()
accuracy = correct_predictions / len(test_targets)

logging.info(f"the test accuracy was {accuracy}")

mean_per_class_acc = mean_per_class_accuracy(
    preds=predictions,
    targets=test_targets
)

logging.info(f"the mean per-class accuracy was {mean_per_class_acc}")
