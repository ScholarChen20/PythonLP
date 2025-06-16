import os

root_dir = "data/hymenoptera_data/train"
target_dir = "bees_image"
img_path = os.listdir(os.path.join(root_dir,target_dir))
label = target_dir.split('_')[0]
print(label)
out_dir = "bees_label"
for i in img_path:
    file_name = i.split('.jpg')[0]
    print(file_name)
    with open(os.path.join(root_dir,out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)
