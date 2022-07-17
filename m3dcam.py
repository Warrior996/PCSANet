# Import M3d-CAM
from Medcam.medcam import medcam
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import NIH
import pandas as pd
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Init your model and dataloader
# model = ResNet50(N_LABELS=14, isTrained=True).cuda()
CheckPointData = torch.load('results_convnext_nnIsTrainedTrue/checkpoint')
model = CheckPointData['model']

path_image = "../../../data/ChestX-ray14/images"
val_df_path = "labels/grad-cam.csv"
val_df = pd.read_csv(val_df_path)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset_val = NIH(val_df, path_image=path_image, transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize]))
data_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)
size = len(val_df)
print("val_df size :", size)

# Inject model with M3d-CAM
# model = medcam.inject(model, output_dir="attention_maps", save_maps=True)
model = medcam.inject(model, output_dir='attention_maps', backend='gcam', save_maps=True)
# Continue to do what you're doing...
# In this case inference on some new data
# model.eval()
# for i, data in enumerate(data_loader):
#         inputs, labels, item = data
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         true_labels = labels.cpu().data.numpy()
#         batch_size = true_labels.shape
#         output = model(inputs)
