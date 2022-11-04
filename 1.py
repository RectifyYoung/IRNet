import csv
import os
import numpy as np

label_path = "/workspace/server/dataset/image.csv"
img_path = "/workspace/server/dataset/video_frame"
save_path = "/workspace/server/dataset/label.csv"
img_names = os.listdir(img_path)

with open(label_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    r = list(reader)
    file_name = []
    for i in range(1,7944):
        file_name.append(r[i][0])
    # r1 = np.array(r)
    # print(r1.shape)
count=[]
i = 0

for img_name in img_names:
    s = file_name.count(img_name)
    count.append(s)

with open(save_path, "w") as csvfile:
    for img_name in img_names:
    #print(img_name)
        if count[i] == 3:
            ind = file_name.index(img_name)
            print(ind)
            adult_x1 = float(r[ind+1][2])
            adult_y1 = float(r[ind+1][3])
            adult_x2 = float(r[ind+1][4])
            adult_y2 = float(r[ind+1][5])
            object_x1 = float(r[ind+3][2])
            object_y1 = float(r[ind+3][3])
            object_x2 = float(r[ind+3][4])
            object_y2 = float(r[ind+3][5])
            adult_center_x = (adult_x1+adult_x2)/2.0
            adult_center_y = (adult_y1+adult_y2)/2.0
            object_center_x = (object_x1+object_x2)/2.0
            object_center_y = (object_y1+object_y2)/2.0
            inter_x = (adult_center_y+object_center_y)/2.0
            inter_y = (adult_center_x+object_center_x)/2.0
            writer = csv.writer(csvfile)
            writer.writerow([img_name])
        i = i+1
