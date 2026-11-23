"""Tests use in-memory samples and do not rely on CSV files."""
import pandas as pd
from src.data_processing import clean_data, inspect_data
from src.train import make_features

def sample():
 return pd.DataFrame({'record_id':['A','B','B'],'date':['2026-01-01','2026-01-02','2026-01-02'],
 'distance_km':[100,200,200],'weight_kg':[2.5,12500.,12500.],'hub_backlog':[100,200,200],
 'pin_code_tier':[1,2,2],'declared_value_rs':[1000,2000,2000],'transfers':[1,2,2],
 'service':['standard ','express','express'],'destination':['Metro','Tier-2','Tier-2'],
 'days_from_tracking_page':[3.1,4.2,4.2],'delivery_days':[3.,4.,4.]})

def test_duplicates_and_leakage_removed():
 c=clean_data(sample()); assert len(c)==2 and 'days_from_tracking_page' not in c

def test_tier2_weight_converted_and_text_trimmed():
 c=clean_data(sample()); assert c.loc[c.destination=='Tier-2','weight_kg'].iloc[0]==12.5
 assert c.service.tolist()==['standard','express']

def test_inspection_detects_known_data_issues():
 r=inspect_data(sample()); assert r['duplicate_extra_rows']==1
 assert r['leaking_column']=='days_from_tracking_page' and r['unit_mixup']['conversion_factor']==1000

def test_model_features_exclude_id_target_and_leakage():
 x=make_features(clean_data(sample())); assert not {'record_id','delivery_days','days_from_tracking_page'} & set(x.columns)
 assert 'date_ordinal' in x
