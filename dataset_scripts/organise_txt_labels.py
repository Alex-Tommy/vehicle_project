import os


old_dir = '../UAV-benchmark-MOTD_v1.0/GT'
output_dir = '../dataset/labels/all'
if not os.path.exists(output_dir):
    os.makedirs(output_dir,mode=0o777)
IMG_W = 1024
IMG_H = 540

video_label_txts = os.listdir(old_dir)

for each_txt in video_label_txts:    
    if each_txt[-6:] == 'gt.txt':
        # Esempio 'M1006_gt_whole.txt'
        video_name = each_txt[:5]    #'M1006'
        txt_path = old_dir + '/' + each_txt  # '../UAV-benchmark-MOTD_v1.0/GT/M1006_gt_whole.txt'

        # read txt
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line_ls = line.split(',')    
                img_num = line_ls[0]    
                img_six_num = (6-len(img_num))*'0' + str(img_num)    # '001089'

                # trasforma [x1, y1, w, h] in [xc, yc, w, h]
                org_xc = int(line_ls[2]) + int(line_ls[4])/2
                org_yc = int(line_ls[3]) + int(line_ls[5])/2
                org_w = int(line_ls[4])
                org_h = int(line_ls[5])
                
                # normalizza le dimensioni tra [0, 1]
                xc = float(org_xc / IMG_W)
                yc = float(org_yc / IMG_H)
                w = float(org_w / IMG_W)
                h = float(org_h / IMG_H)

                if xc>1:
                    print('oh no!!! xc ', xc)
                if yc>1:
                    print('oh no!!! yc ', yc)
                if w>1:
                    print('oh no!!! w ', w)
                if h>1:
                    print('oh no!!! h ', h)

                # rimuove le label sbagliate (alcune erano molto grandi)  
                wrong_label = False
                if org_w > (IMG_W / 6):
                    if org_h > (IMG_H / 6):
                        wrong_label = True

                # riscrive le bounding boxes in un nuovo file di testo (una immagine corrisponde a un file di testo)
                if not wrong_label:
                    new_txt_path = output_dir + '/' + video_name + '_' + img_six_num + '.txt'
                    with open(new_txt_path, 'a') as wr:
                        bbox = '0' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + '\n'
                        wr.writelines(bbox)
        print(each_txt, ' has been parsed')
print('all txt labels have been saved in: ', output_dir)
