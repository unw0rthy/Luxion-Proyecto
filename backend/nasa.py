"""NASA POWER data access; requests are cached by Streamlit at the UI layer."""
import pandas as pd
import requests

URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'
PARAMS = ['T2M','T2M_MAX','T2M_MIN','PRECTOTCORR']

def get_power(lat, lon, start_year=2018, end_year=2024):
    params = {'parameters': ','.join(PARAMS), 'community':'AG', 'latitude':lat,
              'longitude':lon, 'start':f'{start_year}0101', 'end':f'{end_year}1231',
              'format':'JSON'}
    response = requests.get(URL, params=params,timeout=35)
    response.raise_for_status()
    obj = response.json().get('properties',{}).get('parameter',{})
    if not all(key in obj for key in PARAMS):
        raise ValueError('NASA POWER no devolvió todos los parámetros solicitados.')
    df = pd.DataFrame(obj).rename_axis('date').reset_index()
    df['date'] = pd.to_datetime(df['date'],format='%Y%m%d',errors='coerce')
    for key in PARAMS:
        df[key] = pd.to_numeric(df[key],errors='coerce').replace([-999,-999.0,-9999],float('nan'))
    df = df.dropna(subset=['date','T2M','T2M_MAX','PRECTOTCORR'])
    if len(df) < 90:
        raise ValueError('La respuesta tiene menos de 90 días válidos.')
    return df

def make_synthetic_demo(seed=123):
    """Synthetic *demonstration only*, never label as NASA observation."""
    import numpy as np
    dates = pd.date_range('2018-01-01','2024-12-31',freq='D')
    rng = np.random.default_rng(seed)
    phase = dates.dayofyear.to_numpy()
    temperature = 16+7*np.sin(2*np.pi*(phase-15)/365)+rng.normal(0,2,len(dates))
    precipitation = rng.choice([0., .2, 2, 6],size=len(dates),p=[.965,.02,.012,.003])
    return pd.DataFrame({'date':dates,'T2M':temperature,'T2M_MAX':temperature+8,
                         'T2M_MIN':temperature-8,'PRECTOTCORR':precipitation})
