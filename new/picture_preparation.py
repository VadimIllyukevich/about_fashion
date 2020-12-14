import shutil
import os


def copy_pictures(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(os.path.join(source_dir, "bag" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "bag"))
        shutil.copy2(os.path.join(source_dir, "boots" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "boots"))
        shutil.copy2(os.path.join(source_dir, "coat" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "coat"))
        shutil.copy2(os.path.join(source_dir, "dress" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "dress"))
        shutil.copy2(os.path.join(source_dir, "pants" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "pants"))
        shutil.copy2(os.path.join(source_dir, "shirt" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "shirt"))
        shutil.copy2(os.path.join(source_dir, "shoes" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "shoes"))
        shutil.copy2(os.path.join(source_dir, "sneakers" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "sneakers"))
        shutil.copy2(os.path.join(source_dir, "sweater" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "sweater"))
        shutil.copy2(os.path.join(source_dir, "t-short" + str(i) + ".jpg"),
                     os.path.join(dest_dir, "t-short"))


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "bag"))
    os.makedirs(os.path.join(dir_name, "boots"))
    os.makedirs(os.path.join(dir_name, "coat"))
    os.makedirs(os.path.join(dir_name, "dress"))
    os.makedirs(os.path.join(dir_name, "pants"))
    os.makedirs(os.path.join(dir_name, "shirt"))
    os.makedirs(os.path.join(dir_name, "shoes"))
    os.makedirs(os.path.join(dir_name, "sneakers"))
    os.makedirs(os.path.join(dir_name, "sweater"))
    os.makedirs(os.path.join(dir_name, "t-short"))


data_dir = 'pictures'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

test_data_portion = 0.2
val_data_portion = 0.2
nb_pictures = 5

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

start_val_data_idx = int(nb_pictures * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_pictures * (1 - test_data_portion))

copy_pictures(0, start_val_data_idx, data_dir, train_dir)
copy_pictures(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_pictures(start_test_data_idx, nb_pictures, data_dir, test_dir)
