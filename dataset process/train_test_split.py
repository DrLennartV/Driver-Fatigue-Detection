import os
import os.path
import shutil
import random

oldpath = 'yawning'
newpath = 'train/yawning'

filenames = random.sample(os.listdir(oldpath), 1446)
for fname in filenames:
    shutil.move(oldpath + '/' + fname, newpath + '/' + fname)
