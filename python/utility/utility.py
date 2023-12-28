# benchFileOut = open("benchFileOut.log","a")
# benchFileOut.write("\nTest_Title\n")
def dot_bench_time(benchFileOut, stime, label):
    import time
    etime = time.time()
    print("{}:".format(label), etime - stime)
    benchFileOut.write("{}: {}\n".format(label, etime - stime))
    return etime
# benchFileOut.flush()
# benchFileOut.close()

# e.g.: profile_model_flops_params(model, inputs=[input_data])
def profile_model_flops_params(model, inputs=[], onnx_path=None):
    import thop
    flops, params = thop.profile(model, inputs=inputs)
    print('1. thop: FLOPs = ' + str(flops/1000**2) + 'M / ' +'Params = ' + str(params/1000**2) + 'M')
    if onnx_path != None:
        print('2. onnx_profile: ')
        import onnx_tool
        onnx_tool.model_profile(onnx_path)
    import torchstat
    print('3. torchstat: ')
    torchstat.stat(model, inputs[0].shape[1:]) # shape without batch

def export_model_to_onnx(model, input_data, name="model.onnx"):
    import torch
    torch.onnx.export(model,             # model being run
              input_data,                # model input (or a tuple for multiple inputs)
              name,                      # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,
              verbose=True,
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
    )
    print("onnx_model exported!")

def print_dataLoader_first(data_loader):
    for data, target in data_loader:
        print(data, data.shape, target.tolist())
        break
    #sys.exit(-1)
