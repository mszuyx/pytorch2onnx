from onnxruntime.transformers.onnx_model import OnnxModel
import onnx

model_name_ = 'ostr_mb3_large_ver2.onnx'
new_model_name_ = 'ostr_mb3_large_ver2_shrinked.onnx'

model = onnx.load(model_name_)
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
onnx_model.save_model_to_file(new_model_name_)