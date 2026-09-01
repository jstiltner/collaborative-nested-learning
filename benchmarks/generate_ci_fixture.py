"""Generate a small, seeded, committed MNIST subset for hermetic CI runs.

CI shouldn't depend on a live torchvision MNIST download (network flakiness, an external
mirror's uptime) or need the full 70k-image dataset to prove the bridge-ablation code path
works. This subsamples a fixed number of real images per digit class from the real MNIST
(already cached locally under ./data from previous full runs) and saves them as a compact,
committed tensor file.

Run once (or whenever the fixture size changes):
    python -m benchmarks.generate_ci_fixture
"""

import torch
from torchvision import datasets

TRAIN_PER_CLASS = 3000
TEST_PER_CLASS = 500
SEED = 0
OUT_PATH = "benchmarks/fixtures/mnist_ci_fixture.pt"


def subsample(dataset, per_class: int, generator: torch.Generator):
    """Pick `per_class` raw uint8 images per digit from dataset.data/.targets.

    Stores raw uint8 pixels (1 byte/pixel), not the normalized float transform —
    ~4x smaller on disk. Normalization is applied at load time instead.
    """
    images, labels = [], []
    for c in range(10):
        idxs = (dataset.targets == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idxs), generator=generator)[:per_class]
        chosen = idxs[perm]
        images.append(dataset.data[chosen])  # uint8, [per_class, 28, 28]
        labels.append(torch.full((len(chosen),), c, dtype=torch.long))
    return torch.cat(images), torch.cat(labels)


def main():
    train_full = datasets.MNIST(root="./data", train=True, download=True)
    test_full = datasets.MNIST(root="./data", train=False, download=True)

    g = torch.Generator().manual_seed(SEED)
    train_images, train_labels = subsample(train_full, TRAIN_PER_CLASS, g)
    test_images, test_labels = subsample(test_full, TEST_PER_CLASS, g)

    import os

    os.makedirs("benchmarks/fixtures", exist_ok=True)
    torch.save(
        {
            "train_images": train_images,
            "train_labels": train_labels,
            "test_images": test_images,
            "test_labels": test_labels,
            "train_per_class": TRAIN_PER_CLASS,
            "test_per_class": TEST_PER_CLASS,
            "seed": SEED,
        },
        OUT_PATH,
    )
    print(f"Saved fixture: {train_images.shape[0]} train, {test_images.shape[0]} test images "
          f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
