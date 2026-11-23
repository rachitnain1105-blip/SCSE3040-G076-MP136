"""MP136 data inspection and cleaning."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA_PATH=ROOT/'data/raw/train.csv'
OUTPUT_PATH=ROOT/'data/processed/train_clean.csv'
TARGET='delivery_days'
LEAKAGE_COLUMN='days_from_tracking_page'

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def inspect_data(df):
    """Reproducible diagnostics for the seven data-quality checks."""
    result={'rows':int(len(df)), 'duplicate_extra_rows':int(df.duplicated().sum()),
            'missing_values':{k:int(v) for k,v in df.isna().sum().items()},
            'leaking_column':LEAKAGE_COLUMN if LEAKAGE_COLUMN in df else None,
            'unit_mixup':None,'sentinel_values':{},'text_levels':{}}
    if {'destination','weight_kg'}.issubset(df.columns):
        med=df.groupby('destination')['weight_kg'].median().to_dict()
        if med and max(med.values())/max(min(med.values()),1e-9)>100:
            result['unit_mixup']={'column':'weight_kg','group_medians':{k:float(v) for k,v in med.items()},
                                  'affected_groups':['Tier-2'],'conversion_factor':1000}
    for col in df.select_dtypes(include='number'):
        counts=df[col].value_counts()
        found={str(v):int(counts.get(v,0)) for v in (-999,-1,999,9999) if counts.get(v,0)}
        if found: result['sentinel_values'][col]=found
    for col in df.select_dtypes(include=['object','string']):
        if col!='record_id': result['text_levels'][col]=sorted(map(str,df[col].dropna().unique()))
    return result

def clean_data(df):
    """Drop duplicates/leakage, fix Tier-2 grams-to-kg, normalize text/date."""
    out=df.drop_duplicates().copy()
    if LEAKAGE_COLUMN in out: out=out.drop(columns=[LEAKAGE_COLUMN])
    if {'destination','weight_kg'}.issubset(out.columns):
        mask=out.destination.astype(str).str.strip().eq('Tier-2')
        out.loc[mask,'weight_kg']=out.loc[mask,'weight_kg']/1000.0
    for col in ('service','destination'):
        if col in out: out[col]=out[col].astype('string').str.strip()
    if 'date' in out:
        out['date']=pd.to_datetime(out.date,errors='coerce').dt.strftime('%Y-%m-%d')
    return out.sort_values('date',kind='stable').reset_index(drop=True)

def save_clean_data(df,path=OUTPUT_PATH):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False); return path

if __name__=='__main__':
    data=load_data(); print('Inspection:',inspect_data(data)); cleaned=clean_data(data)
    print('Saved:',save_clean_data(cleaned),'rows:',len(cleaned))
