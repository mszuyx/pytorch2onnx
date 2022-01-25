from onnxruntime.transformers.onnx_model import OnnxModel
import onnx
import time
import matplotlib.pyplot as plt
import torch
import numpy as np

import onnxruntime
print("onnxruntime version: ", onnxruntime.__version__)

model_name_ = 'ostr_mb3_large_ver2'
new_model_name_ = model_name_ + "_shrinked"

model = onnx.load(model_name_ + '.onnx')
onnx_model=OnnxModel(model)

def has_same_value(val_one,val_two):
  if val_one.raw_data == val_two.raw_data:
    return True
  else:
    return False

count = len(model.graph.initializer)
same = [-1] * count
for i in range(count - 1):
    if same[i] >= 0:
        continue
    for j in range(i+1, count):
        # print(model.graph.initializer[i].name)
        # print(model.graph.initializer[j].name)
        # print("====================================")
        if has_same_value(model.graph.initializer[i], model.graph.initializer[j]):
            print("duplicated weight found!")
            same[j] = i

for i in range(count):
    if same[i] >= 0:
        onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name, model.graph.initializer[same[i]].name)

onnx_model.update_graph()
onnx_model.save_model_to_file(new_model_name_ + '.onnx')

# ======================================================================================================

onnx_model = onnxruntime.InferenceSession(model_name_ + ".onnx")
onnx_model_saved = onnxruntime.InferenceSession(new_model_name_ + ".onnx")

# Input to the model
x1 = torch.randn(1, 3, 256, 256).float() #, requires_grad=True
x2 = torch.randn(1, 3, 256, 256).float() #, requires_grad=True
if torch.cuda.is_available():
    x1 = x1.cuda()
    x2 = x2.cuda()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
onnx_inputs = {onnx_model.get_inputs()[0].name: to_numpy(x1), onnx_model.get_inputs()[1].name:to_numpy(x2)}
onnx_outs = onnx_model.run(None, onnx_inputs)
t = time.time()
onnx_outs_new = onnx_model_saved.run(None, onnx_inputs)
elapsed = time.time() - t

# compare ONNX Runtime and PyTorch results
print("Are both model outputs similar? : ",np.allclose(onnx_outs[0][0], onnx_outs_new[0][0], rtol=1e-03, atol=1e-05))
print("Saved model inference time: "+str(elapsed))

fig = plt.figure(0)
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(np.transpose(onnx_outs[0][0],(1, 2, 0)))
ax.set_title('Original model output')
ax.axis('off')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(np.transpose(onnx_outs_new[0][0],(1, 2, 0)))
ax.set_title('Saved model output')
ax.axis('off')
plt.show()