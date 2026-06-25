CLASSNAMES = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "cifar100": ["apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm"],
}

_TEMPLATE_STRINGS = {
    "cifar10": ["a photo of a {c}.", "a blurry photo of a {c}.", "a low contrast photo of a {c}.", "a good photo of a {c}.", "a photo of the {c}."],
    "cifar100": ["a photo of a {c}.", "a blurry photo of a {c}.", "a low contrast photo of a {c}.", "a good photo of a {c}.", "a photo of the {c}."],
}

TEMPLATES = {
    k: [(lambda c, s=s: s.format(c=c)) for s in v]
    for k, v in _TEMPLATE_STRINGS.items()
}


def build_dataset(name, root, preprocess, train=False):
    import torchvision.datasets as tvd
    if name == "cifar10":
        return tvd.CIFAR10(root, train=train, download=True, transform=preprocess)
    if name == "cifar100":
        return tvd.CIFAR100(root, train=train, download=True, transform=preprocess)
    raise ValueError(name)


def build_loaders(names, root, preprocess, batch_size=256, workers=2):
    from torch.utils.data import DataLoader
    loaders = {}
    for n in names:
        ds = build_dataset(n, root, preprocess)
        loaders[n] = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    return loaders
