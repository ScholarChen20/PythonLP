from PIL import Image
from torch.utils.data import Dataset
import os

class Mydata(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(root_dir,label_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

rootdir = "data/hymenoptera_data/train"
ant_label = "ants_image"
bees_label = "bees_image"
ann_dataset = Mydata(rootdir,ant_label)
bees_dataset = Mydata(rootdir,bees_label)
