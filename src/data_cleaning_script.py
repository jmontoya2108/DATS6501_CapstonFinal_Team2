"""
PROCUREMENT DATA CLEANING SCRIPT
================================

This script demonstrates the complete data cleaning process applied to the 
Datasets Procurement.xlsx file.

Author: Data Cleaning Process Documentation
Date: February 8, 2026

Original Dataset: data/Datasets Procurement.xlsx (9,894 rows in Data02.07)
Cleaned Dataset: data/Datasets Procurement_Cleaned.xlsx (7,518 rows in Data02.07)

Total Rows Removed: 2,376 (24.0% reduction)
Financial Impact: ~$53.3M in charges/fees removed
"""
#%%
import pandas as pd
import shutil
from datetime import datetime
from openpyxl import load_workbook

print("=" * 100)
print("PROCUREMENT DATASET CLEANING PROCESS")
print("=" * 100)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# ==================================================================================================
# STEP 1: LOAD ORIGINAL DATASET
# ==================================================================================================
print("\n" + "=" * 100)
print("STEP 1: LOADING ORIGINAL DATASET")
print("=" * 100)

original_file = 'data/Datasets Procurement.xlsx'
cleaned_file = 'data/Datasets Procurement_Cleaned.xlsx'

# Create a backup of original file (optional)
backup_file = 'data/Datasets Procurement_BACKUP.xlsx'
print(f"\nCreating backup: {backup_file}")
shutil.copy2(original_file, backup_file)
print("✓ Backup created")

# Load the original data
print(f"\nLoading original dataset: {original_file}")
excel_file = pd.ExcelFile(original_file)
print(f"Available sheets: {excel_file.sheet_names}")

df_data_original = pd.read_excel(original_file, sheet_name='Data02.07')
df_receipt_original = pd.read_excel(original_file, sheet_name='Receipt data 02.07')

print(f"\nOriginal Data02.07: {len(df_data_original)} rows × {len(df_data_original.columns)} columns")
print(f"Original Receipt data: {len(df_receipt_original)} rows × {len(df_receipt_original.columns)} columns")


#%%
# ==================================================================================================
# STEP 2: REMOVE INSTALLATION ROWS
# ==================================================================================================
print("\n" + "=" * 100)
print("STEP 2: REMOVING INSTALLATION ROWS FROM Data02.07")
print("=" * 100)

df_data = df_data_original.copy()

print(f"\nStarting rows: {len(df_data)}")
print("\nAnonymizing PurchaseName values...")

if 'PurchaseName' in df_data.columns:
    purchase_names = df_data['PurchaseName'].dropna().astype(str).str.strip()
    unique_purchase_names = pd.Series(purchase_names.unique())
    supplier_lookup = {
        purchase_name: f"Supplier {idx}"
        for idx, purchase_name in enumerate(unique_purchase_names, start=1)
    }

    df_data['PurchaseName'] = (
        df_data['PurchaseName']
        .astype('string')
        .str.strip()
        .map(supplier_lookup)
        .astype('string')
    )

    print(f"Anonymized {len(supplier_lookup)} unique PurchaseName values")
else:
    print("PurchaseName column not found; skipping supplier anonymization")

print("\nSearching for rows where ItemCode contains 'INSTALLATION'...")

# Find rows with INSTALLATION in ItemCode
installation_mask = df_data['ItemCode'].astype(str).str.contains('INSTALLATION', case=False, na=False)
installation_rows = df_data[installation_mask]

print(f"Found: {len(installation_rows)} rows with 'INSTALLATION'")
print(f"\nUnique ItemCode values:")
for code in installation_rows['ItemCode'].unique():
    count = (installation_rows['ItemCode'] == code).sum()
    print(f"  - {code}: {count} rows")

# Remove INSTALLATION rows
df_data = df_data[~installation_mask]
print(f"\n✓ Removed {len(installation_rows)} INSTALLATION rows")
print(f"Remaining rows: {len(df_data)}")

#%%
# ==================================================================================================
# STEP 3: REMOVE FREIGHT ROWS
# ==================================================================================================
print("\n" + "=" * 100)
print("STEP 3: REMOVING FREIGHT ROWS FROM Data02.07")
print("=" * 100)

print(f"\nStarting rows: {len(df_data)}")
print("\nSearching for rows where ItemCode contains 'FREIGHT'...")

# Find rows with FREIGHT in ItemCode
freight_mask = df_data['ItemCode'].astype(str).str.contains('FREIGHT', case=False, na=False)
freight_rows = df_data[freight_mask]

print(f"Found: {len(freight_rows)} rows with 'FREIGHT'")
print(f"\nUnique ItemCode values:")
for code in freight_rows['ItemCode'].unique():
    count = (freight_rows['ItemCode'] == code).sum()
    print(f"  - {code}: {count} rows")

print(f"\nFinancial Impact:")
print(f"  - Total ExtensionAmt: ${freight_rows['ExtensionAmt'].sum():,.2f}")
print(f"  - Total InvoicedAmt: ${freight_rows['InvoicedAmt'].sum():,.2f}")

# Remove FREIGHT rows
df_data = df_data[~freight_mask]
print(f"\n✓ Removed {len(freight_rows)} FREIGHT rows")
print(f"Remaining rows: {len(df_data)}")

#%%
# ==================================================================================================
# STEP 4: REMOVE SERVICE/CHARGE ROWS
# ==================================================================================================
print("\n" + "=" * 100)
print("STEP 5: REMOVING SERVICE/CHARGE ROWS FROM Data02.07")
print("=" * 100)

print(f"\nStarting rows: {len(df_data)}")

# Define service/charge terms to remove
service_terms = [
    'materials', 'Travel', 'Tax', 'tariff', 'support', 'repair', 'rental',
    'miscellaneous charge', 'packaging', 'down payment', 'creditmemo', 'labor',
    'long lead funding', 'office trailer', 'restroom trailer', 'discount',
    'drawings', 'engineering', 'expedite charge', 'mobilization',
    'portable restroom', 'Field evaluation', 'fee', 'pallet charge', '*tools', '*setup', '*LONG LEADS FUNDING','DUMPSTERS'
]

print(f"\nSearching for {len(service_terms)} service/charge terms in ItemCode...")

# Create combined mask for all service terms
all_masks = []
term_counts = {}

for term in service_terms:
    mask = df_data['ItemCode'].astype(str).str.contains(term, case=False, na=False, regex=False)
    count = mask.sum()
    if count > 0:
        term_counts[term] = count
        all_masks.append(mask)

# Combine all masks with OR
combined_mask = pd.DataFrame(all_masks).T.any(axis=1)
service_rows = df_data[combined_mask]

print(f"\nFound {len(service_rows)} rows containing service/charge terms:")
for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {term:30s}: {count:4d} rows")

print(f"\nFinancial Impact:")
print(f"  - Total ExtensionAmt: ${service_rows['ExtensionAmt'].sum():,.2f}")
print(f"  - Total InvoicedAmt: ${service_rows['InvoicedAmt'].sum():,.2f}")

# Remove service/charge rows
df_data = df_data[~combined_mask]
print(f"\n✓ Removed {len(service_rows)} service/charge rows")
print(f"Remaining rows: {len(df_data)}")

#%%
print("\n" + "=" * 100)
print("STEP 5: FILLING NULL ItemCodeDesc VALUES")
print("=" * 100)

null_before = df_data['ItemCodeDesc'].isna().sum()
print(f"\nItemCodeDesc null values before: {null_before} ({(null_before/len(df_data)*100):.1f}%)")

itemcode_str = df_data['ItemCode'].astype(str)

mask = (
    df_data['ItemCodeDesc'].isna() &
    (itemcode_str.str[0].str.isdigit() | itemcode_str.str.startswith('HSIU'))
)

rows_to_fill = mask.sum()
print(f"\nRows matching criteria: {rows_to_fill}")

df_data.loc[mask, 'ItemCodeDesc'] = df_data.loc[mask, 'ItemCode']

null_after = df_data['ItemCodeDesc'].isna().sum()
reduction_pct = ((null_before - null_after) / null_before * 100) if null_before else 0

print(f"\nItemCodeDesc null values after: {null_after} ({(null_after/len(df_data)*100):.1f}%)")
print(f"✓ Filled {null_before - null_after} cells")
print(f"Improvement: {reduction_pct:.1f}% reduction in nulls")

#%%
# ==================================================================================================
# STEP 5: FILL NULL ItemCodeDesc VALUES
# ==================================================================================================
print("\n" + "=" * 100)
print("STEP 5: FILLING NULL ItemCodeDesc VALUES")
print("=" * 100)

null_before = df_data['ItemCodeDesc'].isna().sum()
print(f"\nItemCodeDesc null values before: {null_before} ({(null_before/len(df_data)*100):.1f}%)")

print("\nFilling null ItemCodeDesc where:")
print("  - ItemCode starts with a digit (e.g., 1003-42003)")
print("  - OR ItemCode starts with 'HSIU' (e.g., HSIU-1780)")

# Create mask for rows to fill
mask = (df_data['ItemCodeDesc'].isna() & 
        (df_data['ItemCode'].astype(str).str.match(r'^\d') | 
         df_data['ItemCode'].astype(str).str.startswith('HSIU')))

rows_to_fill = mask.sum()
print(f"\nRows matching criteria: {rows_to_fill}")

# Fill the values
df_data.loc[mask, 'ItemCodeDesc'] = df_data.loc[mask, 'ItemCode']

null_after = df_data['ItemCodeDesc'].isna().sum()
print(f"\nItemCodeDesc null values after: {null_after} ({(null_after/len(df_data)*100):.1f}%)")
print(f"✓ Filled {null_before - null_after} cells")
print(f"Improvement: {((null_before - null_after) / null_before * 100):.1f}% reduction in nulls")

#%%
# ===================================================================
# STEP 5B: Fill remaining NULL ItemCodeDesc with ItemCode
# ===================================================================

remaining_before = df_data['ItemCodeDesc'].isna().sum()
print(f"\nRemaining null ItemCodeDesc before fallback fill: {remaining_before}")

# Fill ONLY the remaining nulls
df_data['ItemCodeDesc'] = df_data['ItemCodeDesc'].fillna(df_data['ItemCode'])

remaining_after = df_data['ItemCodeDesc'].isna().sum()
print(f"Remaining null ItemCodeDesc after fallback fill: {remaining_after}")
print(f"✓ Additional filled: {remaining_before - remaining_after}")



#%%
# STEP 8: RENAMING SOME VARIABLES
import pandas as pd

df_data = df_data.rename(columns={
    'ItemCodeDesc': 'Manufacturer Number',
    'PO-Item': 'Description'})

#%%
df_data = df_data.drop(columns=['UDF_REVISION', 'Task'])

#%%
df_data.head()

#%%
# ==================================================================================================
# STEP 9: CHECK FOR MISSING VALUES
# ==================================================================================================
null_summary = pd.DataFrame({
    'Null_Count': df_data.isnull().sum(),
    'Null_Percentage': df_data.isnull().mean() * 100
})

print(null_summary)

#%%
# ==================================================================================================
# STEP 8: SAVE CLEANED DATASET
# ==================================================================================================
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_file = f"data/Datasets_Procurement_Cleaned_{timestamp}.xlsx"

print(f"\nSaving cleaned dataset to: {safe_file}")

with pd.ExcelWriter(safe_file, engine='openpyxl') as writer:
    df_data.to_excel(writer, sheet_name='Data02.07', index=False)

print("✓ Dataset saved successfully")

#%%
df_data.shape

#%%
# ==================================================================================================
# FINAL SUMMARY
# ==================================================================================================
print("\n" + "=" * 100)
print("CLEANING COMPLETE - FINAL SUMMARY")
print("=" * 100)

print("\n📊 DATA CLEANING RESULTS:")
print("\nData02.07 Sheet:")
print(f"  Original rows:         {len(df_data_original):,}")
print(f"  - INSTALLATION rows:   -{len(installation_rows):,}")
print(f"  - FREIGHT rows:        -{len(freight_rows):,}")
print(f"  - Service/charge rows: -{len(service_rows):,}")
print(f"  {'='*30}")
print(f"  Final rows:            {len(df_data):,}")
print(f"  Total removed:         {len(df_data_original) - len(df_data):,} ({((len(df_data_original) - len(df_data)) / len(df_data_original) * 100):.1f}%)")

print("\nReceipt data 02.07 Sheet:")
print(f"  Rows: {len(df_receipt):,} (unchanged)")

print("\n💰 FINANCIAL IMPACT:")
total_removed_extension = installation_rows['ExtensionAmt'].sum() + freight_rows['ExtensionAmt'].sum() + service_rows['ExtensionAmt'].sum()
total_removed_invoiced = installation_rows['InvoicedAmt'].sum() + freight_rows['InvoicedAmt'].sum() + service_rows['InvoicedAmt'].sum()
print(f"  Total ExtensionAmt removed: ${total_removed_extension:,.2f}")
print(f"  Total InvoicedAmt removed:  ${total_removed_invoiced:,.2f}")

print("\n📋 DATA QUALITY IMPROVEMENTS:")
print(f"  ItemCodeDesc completeness:")
print(f"    Before: {((len(df_data_original) - df_data_original['ItemCodeDesc'].isna().sum()) / len(df_data_original) * 100):.1f}%")
print(f"    After:  {((len(df_data) - null_after) / len(df_data) * 100):.1f}%")
print(f"    Improvement: +{((len(df_data) - null_after) / len(df_data) * 100) - ((len(df_data_original) - df_data_original['ItemCodeDesc'].isna().sum()) / len(df_data_original) * 100):.1f}%")

print("\n📁 OUTPUT FILES:")
print(f"  Original (preserved):  {original_file}")
print(f"  Backup:                {backup_file}")
print(f"  Cleaned dataset:       {cleaned_file}")

print("\n✅ DATASET CLEANING COMPLETED SUCCESSFULLY!")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# %%
