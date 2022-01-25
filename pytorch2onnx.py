import numpy as np
import time
import matplotlib.pyplot as plt
import torch.onnx
import onnxruntime
print(onnxruntime.__version__)

do_conversion_ = False


# model_path_ = '/home/ros/OS_TR/log/tcd_alot_dtd_OSnet_mb_large_weighted_bce_LR_0.001/snapshot-epoch_2021-12-03-18:34:12_texture.pth' # Best performance
model_path_ = '/home/ros/OS_TR/log/tcd_alot_dtd_lmt_OSnet_mb_large_ver2_weighted_bce_LR_0.001/snapshot-epoch_2022_01_06_19_03_30_texture.pth'

model_name_ = 'ostr_mb3_large_ver2'

from utils.model import models
model = models['OSnet_mb_large_ver2']()
model.load_state_dict(torch.load(model_path_))
# model = torch.load(model_path_)
model.eval()
if torch.cuda.is_available():
    print('pytorch using CUDA version: ',torch.version.cuda)
    model = model.cuda()

# Input to the model
x1 = torch.randn(1, 3, 256, 256).float() #, requires_grad=True
x2 = torch.randn(1, 3, 256, 256).float() #, requires_grad=True
if torch.cuda.is_available():
    x1 = x1.cuda()
    x2 = x2.cuda()

torch_out = model(x1,x2)

# Export the model
if do_conversion_:
  torch.onnx.export(model,                     # model being run
                    (x1,x2),                         # model input (or a tuple for multiple inputs)
                    model_name_ + ".onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=14,          # the ONNX version to export the model to
                  #   do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input1','input2'],   # the model's input names
                    output_names = ['output'], # the model's output names
                  #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #                 'output' : {0 : 'batch_size'}}
                  )

# onnx_model = onnxruntime.InferenceSession(model_name_ + ".onnx")
onnx_model = onnxruntime.InferenceSession(model_name_ + "_shrinked" + ".onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
onnx_inputs = {onnx_model.get_inputs()[0].name: to_numpy(x1), onnx_model.get_inputs()[1].name:to_numpy(x2)}
t = time.time()
onnx_outs = onnx_model.run(None, onnx_inputs)
elapsed = time.time() - t


# compare ONNX Runtime and PyTorch results
# print(np.testing.assert_allclose(to_numpy(torch_out), onnx_outs[0], rtol=1e-03, atol=1e-05))
print("Are both model outputs similar? : ",np.allclose(to_numpy(torch_out), onnx_outs[0], rtol=1e-03, atol=1e-05))
print("Saved model inference time: "+str(elapsed))

fig = plt.figure(0)
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(torch_out[0].permute(1, 2, 0).data.cpu().numpy())
ax.set_title('Original model output')
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(np.transpose(onnx_outs[0][0],(1, 2, 0)))
ax.set_title('Saved model output')
ax.axis('off')
plt.show()