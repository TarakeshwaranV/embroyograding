import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def create_clean_folds(
    input_csv="Blastocyst labels - labels.csv",
    output_csv="labels_clean.csv",
    n_splits=5,
    random_state=42
):
    print("=" * 65)
    print("Creating Clean, 100% Leak-Free Patient-Stratified Folds")
    print("=" * 65)

    df = pd.read_csv(input_csv)
    print(f"Total rows in raw CSV: {len(df)}")
    
    # Extract patient/embryo identifier (prefix before '_')
    df['patient_id'] = df['filename'].apply(lambda x: str(x).split('_')[0])
    
    # Separate labeled and unlabeled images
    is_labeled = df['grade'].isin(['Grade-1', 'Grade-2', 'Grade-3'])
    labeled_df = df[is_labeled].copy().reset_index(drop=True)
    unlabeled_df = df[~is_labeled].copy().reset_index(drop=True)
    
    print(f"Labeled images: {len(labeled_df)} across {labeled_df['patient_id'].nunique()} unique patients")
    print(f"Unlabeled images: {len(unlabeled_df)} across {unlabeled_df['patient_id'].nunique()} unique patients")
    
    # Stratified Group K-Fold on labeled data
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    labeled_df['fold'] = -1
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(labeled_df, labeled_df['grade'], labeled_df['patient_id'])):
        labeled_df.loc[val_idx, 'fold'] = fold
        
    # Map patient_id -> fold for all labeled patients
    patient_to_fold = labeled_df.groupby('patient_id')['fold'].first().to_dict()
    
    # For unlabeled images, if the patient is in a labeled fold, assign to that fold (to prevent cross-fold leak!)
    # If the patient is completely unlabeled, assign fold = -1 (clean unlabeled pool)
    unlabeled_df['fold'] = unlabeled_df['patient_id'].map(patient_to_fold).fillna(-1).astype(int)
    
    overlap_unlabeled = (unlabeled_df['fold'] != -1).sum()
    pure_unlabeled = (unlabeled_df['fold'] == -1).sum()
    print(f"\nUnlabeled frames from labeled patients (assigned to patient's fold): {overlap_unlabeled}")
    print(f"Purely independent unlabeled images (assigned fold -1): {pure_unlabeled}")
    
    # Combine back into clean dataframe
    clean_df = pd.concat([labeled_df, unlabeled_df], ignore_index=True)
    
    # Verification checks
    print("\n" + "=" * 65)
    print("Verification of Folds Distribution:")
    print("=" * 65)
    labeled_clean = clean_df[clean_df['grade'].isin(['Grade-1', 'Grade-2', 'Grade-3'])]
    print(pd.crosstab(labeled_clean['fold'], labeled_clean['grade'], margins=True))
    
    # Strict Patient Overlap Check
    for f in range(n_splits):
        f_patients = set(clean_df[clean_df['fold'] == f]['patient_id'])
        other_patients = set(clean_df[(clean_df['fold'] != f) & (clean_df['fold'] != -1)]['patient_id'])
        overlap = f_patients.intersection(other_patients)
        assert len(overlap) == 0, f"FATAL: Patient overlap in fold {f}: {overlap}"
        
    print("\nSUCCESS: Patient isolation verified! 0 patient overlap across all folds.")
    
    clean_df.to_csv(output_csv, index=False)
    print(f"Saved clean fold assignments to: {os.path.abspath(output_csv)}")
    print("=" * 65)

if __name__ == "__main__":
    create_clean_folds()
