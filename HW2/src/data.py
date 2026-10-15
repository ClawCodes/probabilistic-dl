"""Dataset protocol and initial 2-D uncertainty benchmark."""
from dataclasses import dataclass
import math
import torch


@dataclass
class DatasetBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_names: list[str]
    feature_shape: tuple[int, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def make_two_moons(seed: int = 7, n_train: int = 500, n_test: int = 1_000, noise: float = 0.12) -> DatasetBundle:
    """Create two interleaving semicircles without requiring scikit-learn."""
    generator = torch.Generator().manual_seed(seed)

    def sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        n0 = n // 2
        angle0 = torch.rand(n0, generator=generator) * math.pi
        angle1 = torch.rand(n - n0, generator=generator) * math.pi
        upper = torch.stack((torch.cos(angle0), torch.sin(angle0)), dim=1)
        lower = torch.stack((1 - torch.cos(angle1), 0.5 - torch.sin(angle1)), dim=1)
        x = torch.cat((upper, lower)) + noise * torch.randn(n, 2, generator=generator)
        y = torch.cat((torch.zeros(n0, dtype=torch.long), torch.ones(n - n0, dtype=torch.long)))
        order = torch.randperm(n, generator=generator)
        return x[order], y[order]

    x_train, y_train = sample(n_train)
    x_test, y_test = sample(n_test)
    return DatasetBundle(x_train, y_train, x_test, y_test, ["moon 0", "moon 1"], (2,))


def load_dataset(name: str, seed: int) -> DatasetBundle:
    if name == "two_moons":
        return make_two_moons(seed=seed)
    if name in {"mnist", "fashion_mnist"}:
        # torchvision downloads to HW2/data on first use. Both are 28x28 image
        # classification benchmarks; Fashion-MNIST is a good final choice.
        from torchvision import datasets, transforms
        dataset_cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        root = __import__("pathlib").Path(__file__).resolve().parents[1] / "data"
        transform = transforms.ToTensor()
        train = dataset_cls(root, train=True, download=True, transform=transform)
        test = dataset_cls(root, train=False, download=True, transform=transform)
        x_train = torch.stack([train[i][0] for i in range(len(train))])
        y_train = torch.tensor(train.targets, dtype=torch.long)
        x_test = torch.stack([test[i][0] for i in range(len(test))])
        y_test = torch.tensor(test.targets, dtype=torch.long)
        names = [str(i) for i in range(10)] if name == "mnist" else ["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
        return DatasetBundle(x_train, y_train, x_test, y_test, names, (1, 28, 28))
    raise ValueError(f"Unknown dataset '{name}'. Add it to src/data.py and this registry.")
