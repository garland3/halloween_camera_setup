# %%
from ml.ml_camera import load_ml_model, get_single_img, model_meta_data, visualize_prediction
import torch

# %%

image = get_single_img()
feature_extractor, model = load_ml_model()
metadata = model_meta_data()

# image = image.resize((150,150))
sz = image.size
image = image.crop((sz[0]//2,int(45*sz[1]//100),sz[0],sz[1]))
image

# %%

# image2

# %%


inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    img_size = torch.tensor([tuple(reversed(image.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    
result = visualize_prediction(image, processed_outputs[0], 0.2, metadata['id2label'])
result

# %%
output_dict = processed_outputs[0]
threshold=0.3
keep = output_dict["scores"] > threshold
boxes = output_dict["boxes"][keep].tolist()
scores = output_dict["scores"][keep].tolist()
labels = output_dict["labels"][keep].tolist()
labels
id2label =  metadata['id2label']
labels2 = [id2label[x] for x in labels]
labels2
# %%
id2label
# %%
if 1 in labels:
    print("found peson")
# %%
labels2
# %%
img_size
# %%
