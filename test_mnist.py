from __future__ import print_function
import os
from os import path

ROOT = path.dirname(__file__)
os.sys.path.insert(0, path.join(ROOT, "python"))

import tvm
from tvm import relay, ir
from tvm.relay import testing
from tvm.mrt.utils import *

from tvm.mrt import runtime
from tvm.mrt import stats, dataset
from tvm.mrt import utils

import sys
import numpy as np

from PIL import Image

"""
mnist begin
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 20, 1)
        self.fc1 = nn.Linear(81, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #output = x
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def MnistMain():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model, "mnist_cnn.pt0")


# set 'True' if need to train mnist first
if True:
    MnistMain()
    print("mnist model saved")
    #sys.exit(0)
"""
mnist end
"""

batch_size = 16
num_class = 10
image_shape = (1, 28, 28)
out_shape = (batch_size, num_class)
data_shape = (batch_size,) + image_shape
def load_model_from_torch() -> (ir.IRModule, ParametersT):
    import torch
    from torchvision import models

    #weights = models.ResNet18_Weights.IMAGENET1K_V1
    #model = models.resnet18(weights=weights)
    model = torch.load("mnist_cnn.pt0", map_location=torch.device('cpu'))
    model = model.eval()
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    input_data = torch.randn(data_shape)
    script_module = torch.jit.trace(model, [input_data]).eval()
    #input_dat.cut_last_level()
    return relay.frontend.from_pytorch(
            script_module, [ ("input", data_shape) ])

mod, params = load_model_from_torch()

mod: tvm.IRModule = mod
func: relay.function.Function = mod["main"]
expr: ir.RelayExpr = func.body

from tvm.mrt.trace import Trace
from tvm.mrt.opns import *
from tvm.mrt.symbol import *
tr = Trace.from_expr(expr, params, model_name="mnist_cnn")
# tr = tr.subgraph(onames=["%1"])
tr.checkpoint()
# tr.print(param_config={ "use_all": True, })

tr.print(short=False)
from tvm.mrt import fuse
from tvm.mrt import op
fuse_tr = tr.checkpoint_transform(
        fuse.FuseTupleGetItem.apply(),
        fuse.FuseDropout.apply(),
        fuse.FuseBatchNorm.apply(),
        fuse.FuseAvgPool2D.apply(),
        tr_name = "fuse",
        # force=True,
        )
# fuse_tr.print(param_config={ "use_all": True, })

from tvm.mrt.calibrate import Calibrator, SymmetricMinMaxSampling

calib_tr = fuse_tr.checkpoint_transform(
        Calibrator.apply(random_config={
            "enabled": True,
            "absmax": 1.0, }),
        print_bf=True, print_af=True,
)
# calib_tr.print()
# print(type(calib_tr.symbol))

from tvm.mrt.rules import slm
from tvm.mrt.quantize import Quantizer

# calib_tr = calib_tr.subgraph(onames=["%5"])
dt_tr = calib_tr.checkpoint_transform(
        SymmetricMinMaxSampling.apply(),
        slm.SymmetricLinearDiscretor.apply(),
        )
# dt_tr.print(short=True)
dt_tr: Trace = dt_tr.checkpoint_transform(
        Quantizer.apply(),
        # print_bf=True, print_af=True,
        # force=True,
)

# TODO(wlt): add symbol extra attrs for name_hint to search
#   in subgraph.
# TODO: extra attrs copy and assign logic.
from tvm.mrt.fixed_point import FixPoint, Simulator
# dt_tr.print(short=True, prefix_layers=20)
# FuseBatchNorm.%1
sim_tr = dt_tr.checkpoint_transform(
        Simulator.apply(),
        # force=True,
        )
# sim_tr.log()
# sim_tr.print(short=True)

qt_tr = dt_tr.checkpoint_transform(
        FixPoint.apply(),
        # print_bf = True, print_af = True,
        # force=True,
)
# qt_tr.log()
qt_tr.print(short=False)

from tvm.mrt.zkml import circom, transformer, model as ZkmlModel

symbol, params = qt_tr.symbol, qt_tr.params
symbol, params = ZkmlModel.resize_batch(symbol, params)
print(">>> 1111 change_name ...\n")
ZkmlModel.simple_raw_print(symbol, params)
symbol, params = transformer.change_name(symbol, params)

# set input as params
symbol_first = ZkmlModel.visit_first(symbol)
print(">>> 2222 ...", symbol_first, symbol_first.is_input(), symbol_first.is_param())
#params[symbol_first.name] = symbol_first
#params[symbol_first.name] = symbol_first.numpy()
input_data = torch.randint(255, image_shape)
params[symbol_first.name] = input_data

print(">>> 3333 model2circom ...\n")
out = transformer.model2circom(symbol, params)
print(">>> Generating circom code ...")
code = circom.generate(out)
input_json = transformer.input_json(params)

output_name = "circom_model_test"
print(">>> Generated, dump to {} ...".format(output_name))
#  print(code)
with open(output_name + ".circom", "w") as f:
    f.write(code)
with open(output_name + ".json", "w") as f:
    f.write(json.dumps(input_json, indent=2))

print(">>> finished. exit sys +1 <<<")
sys.exit(1)
