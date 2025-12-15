import h5py

file_path = 'SMAP/SPL4SMGP/SMAP/SMAP_L4_SM_gph_20171231T223000_Vv8010_001.h5'  # 替换为你具体的文件名

with h5py.File(file_path, 'r') as f:
    def printname(name):
        print(name)
    f.visit(printname)
