import os
import shutil


old_dir = '../UAV-benchmark-M'
output_dir = '../dataset/images/all'
if not os.path.exists(output_dir):
    os.makedirs(output_dir,mode=0o777)


folder_names = os.listdir(old_dir)

for folder in folder_names:
    folder_path = old_dir + '/' + folder    
    img_filename_ls = os.listdir(folder_path)   

    for img_filename in img_filename_ls:
        old_img_path = old_dir + '/' + folder + '/' + img_filename
        
        output_img_path = output_dir + '/' + folder + '_' + img_filename[-10:]

        shutil.copyfile(old_img_path, output_img_path)

    print('image folder copy finished: ', folder)
print('all images has been copied into: ', output_dir)
