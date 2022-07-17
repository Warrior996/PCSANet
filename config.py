import os
import torchvision.transforms as transforms
#config for training
config = {
            'CKPT_PATH': 'model/',
            'log_path':  'log/',
            'img_path': 'imgs/',
            'CUDA_VISIBLE_DEVICES': '0', #"0,1,2,3,4,5,6,7"
            'MAX_EPOCHS': 64, 
            'BATCH_SIZE': 16, 
            'TRAN_SIZE': 256,
            'TRAN_CROP': 224,
            'nnIsTrained': True,  # select 'True', 'False'
            'Segmentation': True,  # select 'True', 'False'
            'num_classes': 14,
            'workers': 4,
            'ModelType': 'densenet121',   # select 'resnet50', 'densenet121', 'convnext_base', 'Resume'
         } 

#config for dataset
PATH_TO_IMAGES_DIR = '../../../data/ChestX-ray14/images/'
PATH_TO_BOX_FILE = 'dataset/fjs_BBox.csv'
PATH_TO_TRAIN_VAL_BENCHMARK_FILE = 'dataset/bm_train_val.csv'
PATH_TO_TEST_BENCHMARK_FILE = 'dataset/bm_test.csv'
# PATH_TO_TRAIN_VAL_BENCHMARK_FILE = 'dataset/bm_train_val_copy.csv'
# PATH_TO_TEST_BENCHMARK_FILE = 'dataset/bm_test_copy.csv'
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
N_CLASSES = len(CLASS_NAMES)
transform_seq_test = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
transform_seq_train = transforms.Compose([
   transforms.Resize((256,256)),
   transforms.CenterCrop(224),
   #transforms.RandomCrop(224),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])