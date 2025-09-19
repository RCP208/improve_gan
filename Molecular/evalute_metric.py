# This function is used to evaluate the physical
# properties of the generated molecules.
# As a result, this function screens out all unsuitable molecules
# (not valid, not new, and with unsuitable physics and SA properties)
# in the final dataframe !df!
# or u can check variable 'total_with_sa3' - it is a sum of interesting molecules.
import pdb
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append('~')
sys.path.append(str(import_path))
sys.path.append(str(import_path)+'/../')
sys.path.append(str(import_path)+'/../classifier')
sys.path.append(str(import_path)+'/../../')
sys.path.append(str(import_path)+'/../../../../')
import statistics
from pipeline.classifier import create_sdf_file,Classifier
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from typing import List

path_to_generated_mols_csv = '/home/user/Projects/GEMCODE-main/pipeline/generated_coformers/mols_GAN2_CN1C2=C(C(=O)N(C1=O)C)NC=N2.csv'
generated_mols = pd.read_csv(path_to_generated_mols_csv)['generated_coformers']


def check_self_diversity(smiles:List[str]):
    fpgen = AllChem.GetRDKitFPGenerator()
    self_scores = []
    gen_fp = [fpgen.GetFingerprint(mol) for mol in [Chem.MolFromSmiles(i) for i in smiles]]
    if len(gen_fp)==1:
        return [0]
    for i,mol in enumerate(gen_fp):
        self_scores.append(1-max(DataStructs.BulkTanimotoSimilarity(mol, gen_fp[:i] + gen_fp[i+1 :])))
    return self_scores

def check_chem_valid(smiles:List[str]):
    """Check smiles for chemical validity and return only valid molecules.

    Args:
        smiles (List[str]): Molecules Smiles strings

    Returns:
        List[str]: Molecules Smiles strings
    """
    generated_coformers_clear = []
    for smiles_mol in smiles:
            if smiles_mol=='':
                continue
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue
            generated_coformers_clear.append(smiles_mol)
    generated_coformers_clear = [Chem.MolToSmiles(Chem.MolFromSmiles(str(i))) for i in generated_coformers_clear]
    validaty = len(generated_coformers_clear) / len(smiles)
    # generated_coformer = pd.DataFrame(list(zip([drug] * len(generated_coformers_clear), generated_coformers_clear)),
    #                                   columns=['drug', 'generated_coformers'])
    # path = f'pipeline/generated_coformers/all_valid_mols_{drug}.csv'
    # generated_coformer.to_csv(path)
    return generated_coformers_clear

def check_novelty_mol_path(
        train_dataset_path: str,
        gen_data: list,
        train_col_name: str,
        gen_col_name: str,
        gen_len: int ):
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data: gen molecules.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    # train_df = pd.read_csv(train_dataset_path, header=None)
    # train_d = train_df.iloc[1:, 0]  # 获取所有行的第一列
    # train_d = pd.read_csv(train_dataset_path)[train_col_name]
    train_d = pd.read_csv(train_dataset_path, header=None, usecols=[0])[0]
    gen_d = pd.DataFrame(gen_data,columns=[gen_col_name])
    # pdb.set_trace()
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()/len(gen_d)
    total_len_gen = len(gen_d[gen_col_name])
    #gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d.drop_duplicates())
    novelty =( len(gen_d[gen_col_name].drop_duplicates())-gen_d[gen_col_name].drop_duplicates().isin(train_d).sum() )/ len_gen * 100
    print('Generated molecules consist of',novelty, '% unique new examples',
          '\t',
          f'duplicates: {duplicates}')
    return novelty,duplicates

def check_metrics(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str) ->str:
    """Function for evaluate diversity and new,valid,duplicated molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data_path: Path to csv gen dataset.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    # train_df = pd.read_csv(train_dataset_path, header=None)
    # train_d = train_df.iloc[1:, 0]  # 获取所有行的第一列
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data_path)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    total_len_gen = len(gen_d[gen_col_name])
    gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d)
    gen_d['diversity'] = gen_d[gen_col_name].apply(check_self_diversity)
    mean_diversity= gen_d['diversity'].mean()

    print('Generated molecules consist of',(len_gen-train_d.isin(gen_d).sum())/len_gen*100, '% new examples',
          '\t',f'{len_gen/total_len_gen*100}% valid molecules generated','\t',
          f'duplicates, {duplicates}, mean diversity is {mean_diversity}')

def check_metrics_and_filter(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str,
        filter_conditions: bool = True) -> pd.DataFrame:
    """
    Function for evaluating new/valid/duplicate molecules with optional filtering.
    
    Parameters:
        train_dataset_path: Path to training CSV
        gen_data_path: Path to generated CSV
        train_col_name: Name of molecule column in train dataset
        gen_col_name: Name of molecule column in generated dataset
        filter_conditions: Whether to apply additional filtering
    
    Returns:
        Filtered DataFrame and prints statistics
    """
    # Load data
    # train_d = pd.read_csv(train_dataset_path)[train_col_name]
    train_d = pd.read_csv(train_dataset_path, header=None, usecols=[0])[0]
    # train_df = pd.read_csv(train_dataset_path, header=None)
    # train_d = train_df.iloc[1:, 0]  # 获取所有行的第一列
    gen_d = pd.read_csv(gen_data_path)
    total_initial = len(gen_d)
    
    # Remove duplicates in generated data
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    gen_d = gen_d.drop_duplicates(subset=gen_col_name, keep='first')
    valid_mols = check_chem_valid(gen_d[gen_col_name])
    gen_d = gen_d[~gen_d[gen_col_name].isin(train_d)]
    valid_mols = gen_d[gen_d[gen_col_name].isin(valid_mols)]
    # gen_d = gen_d[new_mols]
    drug = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'
    classification = Classifier()
    df = classification.clf_results(drug, valid_mols[gen_col_name],properties=['unobstructed', 'orthogonal_planes', 'h_bond_bridging'])
    
    if filter_conditions:
        filtered_df = df[
            (df['unobstructed'] == 1) & 
            (df['orthogonal_planes'] == 1) & 
            (df['h_bond_bridging'] == 0)
        ]
        print(f"\nAfter additional filtering: {len(filtered_df)} molecules")
        
        print(f"Filter pass rate: {(len(filtered_df)/total_initial)*100:.2f}%")
        #return filtered_df

    # Keep only valid molecules

    
    
    # Find novel molecules (not present in training set)

    
    # Calculate statistics
    total_after_processing = len(filtered_df)
    filtered_df['diversity'] = check_self_diversity(filtered_df[gen_col_name])
    mean_diversity= filtered_df['diversity'].mean()
    print(f"Initially generated: {total_initial} molecules")
    print(f"After processing: {total_after_processing} molecules")
    print(f"Duplicates removed: {duplicates}")
    print(f"Diversity: {mean_diversity}")
    print(f"Target molecules: {(len(filtered_df)/total_initial)*100:.2f}%")

    # Apply additional filtering if requested

    
    return filtered_df


if __name__=='__main__':
    # Example usage

    filtered_data = check_metrics_and_filter(
        train_dataset_path='/home/user/Projects/GEMCODE-main/generative_models/GAN/CCDC_fine-tuning/data/database_CCDC.csv',
        gen_data_path="/home/user/Projects/GEMCODE-main/pipeline/generated_coformers/mols_GAN2_CN1C2=C(C(=O)N(C1=O)C)NC=N2.csv",
        train_col_name="0.1",
        gen_col_name="generated_coformers"
    )

    # Save results
    filtered_data.to_csv("final_filtered.csv", index=False)
    check_novelty_mol_path(
        train_dataset_path='/home/user/Projects/GEMCODE-main/generative_models/GAN/CCDC_fine-tuning/data/database_CCDC.csv',
        gen_data=generated_mols,
        train_col_name="0.1",
        gen_col_name="generated_coformers",
        gen_len=5712
    )
    # drug = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'
    # print(f'validaty:{check_chem_valid(generated_mols)}')

# target: 1.29 unique:94.4853 duplicate:0.4288 diversity:0.5070 validaty:0.9517
# +0.1 clc
# target: 1.45 unique:96.0412 duplicate:0.3205 diversity:0.5286 validaty:0.926
# +0.1 clc + 0.1 mse
# target: 1.33 unique:94.5721 duplicate:0.3865 diversity:0.4532 validaty:0.9528
# +1to0.1 clc + 0.1 mse
# target: 1.79 unique:95.1436 duplicate:0.3349 diversity:0.5578 validaty:0.9662
# +1to0.1 clc + 0.1to0 mse
# target:1.40 unique:94.8151 duplicate:0.3809 diversity:0.5124 validaty:0.9719
# +1to0.1 clc + 0.1to0.2 mse
# target:1.59 unique:95.0772 duplicate:0.3784 diversity:0.4812 validaty:0.9671
