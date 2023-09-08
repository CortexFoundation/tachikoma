# benchFileOut = open("benchFileOut.log","a")
# benchFileOut.write("\nTest_Title\n")
def dotBenchTime(benchFileOut, stime, label):
    import time
    etime = time.time()
    print("{}:".format(label), etime - stime)
    benchFileOut.write("{}: {}\n".format(label, etime - stime))
    return etime
# benchFileOut.flush()
# benchFileOut.close()

def export_model_to_onnx():
    import torch
    torch.onnx.export(model,             # model being run
              input_data,                # model input (or a tuple for multiple inputs)
              "mnist_cnn.onnx",          # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,
              verbose=True,
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
    )
    print("onnx_model exported!")

