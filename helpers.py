#import settings
import glob
import datetime
import os
import sys
import numpy as np
import cv2
from collections import defaultdict
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import math
import pandas
import re
from PIL import Image, ImageDraw
from scipy import misc
import dicom

def buildHUGDataset(INPUT_FOLDER,list_patients,isPrint):
    entries  = []
    for PID in list_patients:
        if isPrint: print('Patient : ', PID)

        patient_dir =  os.path.join(INPUT_FOLDER, str(PID))
        listscanfile = [name for name in os.listdir(patient_dir) if name.lower().find('.dcm', 0) > 0]
        textfiles_array = [name for name in os.listdir(patient_dir) if name.find('.txt') > 0]
        if len(textfiles_array) != 1:
            print("more than one text file present, please check data directory of patient! ", PID)
            continue
        else:
            text_file = textfiles_array[0]
        f = open(os.path.join(patient_dir, text_file))
        file = f.read()

        spaceing_line = re.compile("SpacingX:.*").findall(file)
        PIXEL_SPACING = float(spaceing_line[0].split(': ')[-1])

        pos_slices = set([])
        label_lines = re.compile("label: .+").findall(file)
        labels = re.compile("label: .+").split(file)
        for a in range(len(label_lines)): # for every label in the annotation file
            if isPrint: print('\t',label_lines[a])
            label_name = label_lines[a].split(' ')[1]

            #divied the label sections into the corrisponding slices
            slice_lines = re.compile("slice_number: [0-9]+").findall(labels[a+1])
            slices = re.compile("slice_number: [0-9]+").split(labels[a+1])

            for i in range(len(slice_lines)): # for every slice
                slice_number = int(slice_lines[i].split(' ')[1]) # get slice number
                if isPrint: print('\t\tslice: ',slice_number)
                pos_slices.add(slice_number)
                #divied the slice section into contours
                contours_lines = re.compile("nb_points_on_contour: [0-9]+").findall(slices[i+1])
                contours         = re.compile("nb_points_on_contour: [0-9]+").split(slices[i+1])
                for j in range(len(contours_lines)): # for every contour section
                    contour_number = int(contours_lines[j].split(' ')[1]) # get the number of points
                    contour_points = contours[j+1].split('\n')
                    contour_coordiantes = []
                    if isPrint: print('\t\t\tROI: ', contour_number, 'points')
                    for point in contour_points:
                        coordiantes = point.split()
                        if len(coordiantes) == 2:
                            pointx = float(coordiantes[0])/PIXEL_SPACING
                            pointy = float(coordiantes[1])/PIXEL_SPACING
                            contour_coordiantes.append((pointx,pointy))
                    entry = [PID,slice_number,label_name,contour_coordiantes]
                    entries.append(entry)

        if isPrint: print('\tlabel: none ')
        snumber = str()
        for i in range(1,len(listscanfile)+1):
            if i not in pos_slices:
                entries.append([PID,i,'none',[]])
                snumber = snumber +  str(i) + " "
        if isPrint: print('\t\tslices: ',snumber)


    dataset = pandas.DataFrame(entries,columns=["patient","slice","label","ROI"])
    return dataset

def createData(dataset,list_patients,l2c,raw_date_path,output_path,display_min = 0,display_max = 4096):
    for PID in list_patients:
        print('Patient : ', PID)
        patient_dir =  os.path.join(raw_date_path, str(PID))
        listscanfile = [name for name in os.listdir(patient_dir) if name.lower().find('.dcm', 0) > 0]

        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(os.path.join(output_path,'input'))
            os.makedirs(os.path.join(output_path,'output'))
        
        for s in range(len(listscanfile)):
            dicom_file_URL = os.path.join(patient_dir, listscanfile[s])
            
            try:
                sl = dicom.read_file(dicom_file_URL)
                img = get_slice_hu(sl)
            except ValueError:
                print(f'error reading file{dicom_file_URL}')


            im_input = Image.fromarray(img)
            im_output, isNone = drawSlice(dataset,PID,s+1,l2c)

            file_name = listscanfile[s].split('.')[0].split('-')[-1]
            file_name = 'patient_' + str(PID) + '_' + file_name
            

            if(isNone):
                file_path_input = output_path + "input/" + file_name + "_neg.png"
                file_path_output = output_path + "output/" + file_name + "_neg.png"
            else:
                file_path_input = output_path + "input/" + file_name + "_pos.png"
                file_path_output = output_path + "output/" + file_name + "_pos.png"

            misc.imsave(file_path_input, im_input)
            misc.imsave(file_path_output, im_output)

def drawSlice(dataset,pid,s,l2c,size=[512,512],color=0):
    # s: slice numbers
    # pid: patient id
    # create empty image for slice (s) and draw on it the corresponing polygons if exist 
    patient_df = dataset[(dataset['patient']==pid) & (dataset['label']!= 'none') ]

    new_img = Image.new("RGB", size, color=(0,0,0))
    draw = ImageDraw.Draw(new_img)

    isNone = False
    if s in patient_df['slice'].unique():
        slices_df = patient_df[patient_df['slice']==s]

        for idx, row in slices_df.iterrows():
            roi = row['ROI']
            draw.polygon(roi, fill = l2c[row['label']])
        del draw
    else:
        isNone = True
    im_arr = np.asarray(new_img)
    return im_arr, isNone

def createColoredMask(im,color):
    #convert a 2D mask image to a RGB image 
    mask = np.stack((im,)*3,-1)
    mask[...,0][mask[...,0]>0] = color[0]
    mask[...,1][mask[...,1]>0] = color[1]
    mask[...,2][mask[...,2]>0] = color[2]
    return mask

def get_slice_hu(slice):
    image = slice.pixel_array
    image = image.astype(np.int16)
    image[image == -2000] = 0
    
    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image+= np.int16(intercept)

    return np.array(image, dtype=np.int16)

def showSlicePrediction(inp,out,prd,alpha=0.4,figsize=(15,7)):
    im = Image.open(inp)
    im_o = Image.open(out)
    im_p = Image.open(prd)
    data = np.asarray( im )
    data_o = np.asarray( im_o )
    data_p = np.asarray( im_p )
    
    f, plots = plt.subplots(1,2,figsize=figsize)
    plots[0].axis('off')
    plots[0].set_title("ground truth")
    plots[0].imshow(data,cmap=plt.cm.bone)
    plots[0].imshow(data_o, alpha=alpha);
    
    plots[1].axis('off')
    plots[1].set_title("prediction")
    plots[1].imshow(data,cmap=plt.cm.bone)
    plots[1].imshow(data_p, alpha=alpha);