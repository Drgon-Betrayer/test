#!/usr/bin/env python
import os
import  shutil

path="/home/sgy/Window_Image"
path_color="/home/sgy/Window_Image/color_img"
path_depth="/home/sgy/Window_Image/depth_img"

def CopyColorFile(colordir):
    filelist=os.listdir(colordir)
    for files in filelist:
        newfile=path_color+"/"+files
        oldfile=colordir+"/"+files
        shutil.move(oldfile,newfile)

def CopyDepFile(depdir):
    filelist=os.listdir(depdir)
    for files in filelist:
        newfile=path_depth+"/"+files
        oldfile=depdir+"/"+files
        shutil.move(oldfile,newfile)

def check(path):
    dirlist=os.listdir(path)
    for dirs in dirlist:
        dirs_full=path+"/"+dirs
        if os.path.isfile(dirs_full):
            continue
        elif dirs=="color":
            CopyColorFile(dirs_full)
        elif dirs=="depth":
            CopyDepFile(dirs_full)
        else:
            check(dirs_full)     

if __name__=="__main__":
    check(path)
