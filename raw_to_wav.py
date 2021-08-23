import os


ori_dir='/data/Dataset/sitec/'
dest_dir='/data/Dataset/sitec/wav/'

ff='/data/Dataset/sitec/Dict01/cln/female/fcb1khh01s184/set184071_cln.raw'

with open(ff, "rb") as raw_f:
    data=raw_f.read()
    print(data)