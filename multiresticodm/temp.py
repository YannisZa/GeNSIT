import os
import shutil
from glob import glob
root = '../data/outputs/cambridge_work_commuter_lsoas_to_msoas/exp2/NonJointTableSIM_NN_SweepedNoise__31_10_2023_09_44_49/samples/'
# for i in range(1,11112):
#     directory = os.path.join(root,f"seed_{i}")
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

for sig in ['low','high','variable']:
    for i in range(1,11112):
        for file in ['data.h5','outputs.log','metadata.json']:
            old_directory = os.path.join(root,f"sigma_{sig}",f"seed_{i}",file)
            if not os.path.exists(os.path.join(root,f"seed_{i}",f"sigma_{sig}")):
                os.makedirs(os.path.join(root,f"seed_{i}",f"sigma_{sig}"))
            new_directory = os.path.join(root,f"seed_{i}",f"sigma_{sig}",file)
            shutil.move(old_directory,new_directory)