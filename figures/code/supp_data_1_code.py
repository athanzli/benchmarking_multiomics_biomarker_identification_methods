import pandas as pd
import glob
import os

folder = "/home/athan.li/eval_bk/result/RRA_results"

csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

csv_files = [
 '/home/athan.li/eval_bk/result/RRA_results/survival_BRCA(CNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_BRCA(DNAm).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_BRCA(SNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_BRCA(mRNA).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_BRCA(miRNA).csv',

 '/home/athan.li/eval_bk/result/RRA_results/survival_LUAD(CNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_LUAD(DNAm).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_LUAD(SNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_LUAD(mRNA).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_LUAD(miRNA).csv',
 
 '/home/athan.li/eval_bk/result/RRA_results/survival_COADREAD(CNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_COADREAD(DNAm).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_COADREAD(SNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_COADREAD(mRNA).csv',
 '/home/athan.li/eval_bk/result/RRA_results/survival_COADREAD(miRNA).csv',

 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Cisplatin-BLCA(CNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Cisplatin-BLCA(DNAm).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Cisplatin-BLCA(SNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Cisplatin-BLCA(mRNA).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Cisplatin-BLCA(miRNA).csv',

 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Temozolomide-LGG(CNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Temozolomide-LGG(DNAm).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Temozolomide-LGG(SNV).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Temozolomide-LGG(mRNA).csv',
 '/home/athan.li/eval_bk/result/RRA_results/drug_response_Temozolomide-LGG(miRNA).csv',
 ]

SHEET_NAME_STANDARD_MAPPING = {
    'survival_BRCA(CNV)' : 'Survival BRCA CNV',
    'survival_BRCA(DNAm)' : 'Survival BRCA DNAm',
    'survival_BRCA(SNV)' : 'Survival BRCA SNV',
    'survival_BRCA(mRNA)' : 'Survival BRCA mRNA',
    'survival_BRCA(miRNA)' : 'Survival BRCA miRNA',
    'survival_LUAD(CNV)' : 'Survival LUAD CNV',
    'survival_LUAD(DNAm)' : 'Survival LUAD DNAm',
    'survival_LUAD(SNV)' : 'Survival LUAD SNV',
    'survival_LUAD(mRNA)' : 'Survival LUAD mRNA',
    'survival_LUAD(miRNA)' : 'Survival LUAD miRNA',
    'survival_COADREAD(CNV)' : 'Survival COADREAD CNV',
    'survival_COADREAD(DNAm)' : 'Survival COADREAD DNAm',
    'survival_COADREAD(SNV)' : 'Survival COADREAD SNV',
    'survival_COADREAD(mRNA)' : 'Survival COADREAD mRNA',
    'survival_COADREAD(miRNA)' : 'Survival COADREAD miRNA',
    'drug_response_Cisplatin-BLCA(CNV)' : 'Drug Res. Cisplatin BLCA CNV',
    'drug_response_Cisplatin-BLCA(DNAm)' : 'Drug Res. Cisplatin BLCA DNAm',
    'drug_response_Cisplatin-BLCA(SNV)' : 'Drug Res. Cisplatin BLCA SNV',
    'drug_response_Cisplatin-BLCA(mRNA)' : 'Drug Res. Cisplatin BLCA mRNA',
    'drug_response_Cisplatin-BLCA(miRNA)' : 'Drug Res. Cisplatin BLCA miRNA',
    'drug_response_Temozolomide-LGG(CNV)' : 'Drug Res. Temoz. LGG CNV',
    'drug_response_Temozolomide-LGG(DNAm)' : 'Drug Res. Temoz. LGG DNAm',
    'drug_response_Temozolomide-LGG(SNV)' : 'Drug Res. Temoz. LGG SNV',
    'drug_response_Temozolomide-LGG(mRNA)' : 'Drug Res. Temoz. LGG mRNA',
    'drug_response_Temozolomide-LGG(miRNA)' : 'Drug Res. Temoz. LGG miRNA',
}

output_excel = os.path.join(folder, "combined.xlsx")

with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        sheet_name = SHEET_NAME_STANDARD_MAPPING[os.path.splitext(os.path.basename(csv_path))[0]] # NOTE excel sheet names have a max length of 31 chars
        df.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"Done. Wrote {len(csv_files)} sheets to {output_excel}")
