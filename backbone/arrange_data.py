import os
import pandas as pd
from tqdm import tqdm

csv_path = '/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/table/brainseg_test_final.csv'
df = pd.read_csv(csv_path)

root = '/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg'
seg_save = f'{root}/subspace/initial_segmentation'
aug_save = f'{root}/augment/preprocess'
conf_save = f'{root}/subspace/confidence_save'

out_path = '/home/xinyu/Awesome_Database/Normative_modeling/GG_Bond_Seg/subspace/data4train/raw_data'


for index, row in tqdm(df[:].iterrows(), total=len(df[:]), desc="Processing Rows"):
    Site,SubjectID,Session = row['Site'],row['SubjectID'],row['Session']
    group = row['Group']
    Session = str(Session)
    SubjectID = str(SubjectID)
    Site = str(Site)

    conf_img = os.path.join(conf_save,f'{Site}_{SubjectID}_{Session}/nii','image_raw.nii.gz')
    conf_conf = os.path.join(conf_save,f'{Site}_{SubjectID}_{Session}/nii','assigned_label_confidence_qG.nii.gz')
    conf_label = os.path.join(conf_save,f'{Site}_{SubjectID}_{Session}/nii','label_raw.nii.gz')

    new_image = f'{out_path}/{Site}_{SubjectID}_{Session}_img_0000.nii.gz'
    new_image1 = f'{out_path}/{Site}_{SubjectID}_{Session}_img_0001.nii.gz'
    new_label = f'{out_path}/{Site}_{SubjectID}_{Session}_label.nii.gz'

    os.system(f'rsync -av {conf_img} {new_image}')
    os.system(f'rsync -av {conf_conf} {new_image1}')
    os.system(f'rsync -av {conf_label} {new_label}')

    # aug_path = os.path.join(aug_save,Site,SubjectID,Session)
    # seg_path = os.path.join(seg_save,Site,SubjectID,Session)

    # image = os.path.join(aug_path,f'HCP206_2_{Site}_{SubjectID}_{Session}_final.nii.gz')
    # label = os.path.join(aug_path,f'HCP206_2_{Site}_{SubjectID}_{Session}_source_seg_final.nii.gz')

    # new_image = f'{out_path}/{Site}_{SubjectID}_{Session}_img_0000.nii.gz'
    # new_label = f'{out_path}/{Site}_{SubjectID}_{Session}_label.nii.gz'

    # os.system(f'rsync -av {image} {new_image}')
    # os.system(f'rsync -av {label} {new_label}')