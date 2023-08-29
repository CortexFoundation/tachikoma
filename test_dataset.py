from tvm.mrt import dataset_torch

def test_imagenet():
    ds = dataset_torch.TorchImageNet()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_coco():
    ds = dataset_torch.TorchCoco()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_voc():
    ds = dataset_torch.TorchVoc()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_cifar10():
    ds = dataset_torch.TorchCifar10()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_cifar100():
    ds = dataset_torch.TorchCifar100()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_mnist():
    ds = dataset_torch.TorchMnist()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_quickdraw():
    ds = dataset_torch.TorchQuickDraw()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_trec():
    ds = dataset_torch.TorchTrec()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_country211():
    ds = dataset_torch.TorchCountry211()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_dtd():
    ds = dataset_torch.TorchDtd()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_emnist():
    ds = dataset_torch.TorchEmnist()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_fashionmnist():
    ds = dataset_torch.TorchFashionMNIST()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_fgvcaircraft():
    ds = dataset_torch.TorchFgvcaircraft()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_flowers102():
    ds = dataset_torch.TorchFlowers102()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_food():
    ds = dataset_torch.TorchFood101()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_gtsrb():
    ds = dataset_torch.TorchGtsrb()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_kmnist():
    ds = dataset_torch.TorchKmnist()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_lfwpeople():
    ds = dataset_torch.TorchLfwpeople()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_omniglot():
    ds = dataset_torch.TorchOmniglot()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_OxfordIIITPet():
    ds = dataset_torch.TorchOxfordIIITPet()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_rendered():
    ds = dataset_torch.TorchRendered()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_stl10():
    ds = dataset_torch.TorchStl10()
    data, label = ds.next()
    print(data.shape, label.shape)

def test_usps():
    ds = dataset_torch.TorchUsps()
    data, label = ds.next()
    print(data.shape, label.shape)

# test_imagenet()
# test_coco()
# test_voc()
# test_cifar10()
# test_cifar100()
# test_mnist()
# test_quickdraw()
# test_trec()
# test_country211()
# test_dtd()
# test_emnist()
# test_fashionmnist()
# test_fgvcaircraft()
# test_flowers102()
# test_food()
# test_gtsrb()
# test_kmnist()
# test_lfwpeople()
# test_omniglot()
# test_OxfordIIITPet()
# test_rendered()
# test_stl10()
# test_usps()
