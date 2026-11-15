"""Agroguard Esencial: comparar una rotación actual y alternativas bajo el mismo clima.
Los parámetros agronómicos son exploratorios, no recomendaciones validadas.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product
from math import isfinite
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from science_engine import CROPS, crop_season, season_windows
from nasa import get_power, make_synthetic_demo
from soil_rules import evaluate_soil, TEXTURES, DRAINAGES, SALINITIES, ORGANIC, FAMILIES

app=FastAPI(title='Agroguard Esencial', version='1.4.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])

class Request(BaseModel):
    lat:float=Field(-22.91,ge=-90,le=90)
    lon:float=Field(-68.2,ge=-180,le=180)
    source:Literal['demo','nasa']='nasa'
    start:int=Field(2018,ge=2001,le=2025)
    end:int=Field(2024,ge=2001,le=2025)
    current_crop:str='Trigo'
    available_crops:list[str]=Field(default_factory=lambda:list(CROPS))
    sow_month:int=Field(4,ge=1,le=12)
    irrigation_mm:float=Field(350,ge=0,le=3000)
    soil_capacity_mm:float=Field(120,ge=20,le=500)
    warming_c:float=Field(0,ge=0,le=6)
    rain_reduction_pct:float=Field(0,ge=0,le=100)
    soil_priority:int=Field(50,ge=0,le=100)
    soil_texture:Literal['Arenoso','Franco','Arcilloso','Desconocido']='Desconocido'
    drainage:Literal['Bueno','Moderado','Deficiente','Desconocido']='Desconocido'
    salinity:Literal['Baja','Media','Alta','Desconocida']='Desconocida'
    organic_matter:Literal['Baja','Media','Alta','Desconocida']='Desconocida'

@lru_cache(maxsize=32)
def climate(lat,lon,start,end,source):
    return get_power(lat,lon,start,end) if source=='nasa' else make_synthetic_demo()

@app.get('/api/health')
def health(): return {'ok':True,'version':'Agroguard Esencial'}

@app.get('/api/crops')
def crops(): return {'crops':list(CROPS)}

CROP_NOTES = {'Avena': 'Cereal de cobertura; revisar disponibilidad de agua y estación', 'Centeno': 'Cereal de cobertura; útil para explorar protección del suelo', 'Maíz': 'Cultivo de verano; puede requerir bastante agua', 'Sorgo': 'Cereal C4; estudiar adaptación a sequía según variedad', 'Lenteja': 'Leguminosa de grano; evaluar fijación de N y drenaje', 'Garbanzo': 'Leguminosa de grano; verificar humedad en floración', 'Arveja': 'Leguminosa de grano; evaluar temperaturas y estación', 'Poroto': 'Leguminosa; requiere evaluación de heladas y agua', 'Girasol': 'Oleaginosa; raíces profundas según condiciones del suelo', 'Canola': 'Brassicácea; alternar familias para gestionar enfermedades', 'Papa': 'Tubérculo; revisar sanidad, riego y salinidad', 'Rábano forrajero': 'Cultivo de cobertura; no supone cosecha comercial', 'Vicia': 'Leguminosa de cobertura; biomasa y fijación de N variables'}

@app.get('/api/crop-catalog')
def crop_catalog():
    return {'crops':[{'name':name,'family':FAMILIES[name],
             'role':'Leguminosa (posible fijación biológica de nitrógeno)' if data['legume'] else 'No leguminosa',
             'note':CROP_NOTES.get(name,'Cultivo de referencia; verificar condiciones y variedad local'),
             'parameters':'Parámetros numéricos exploratorios, no validados'}
             for name,data in CROPS.items()],
             'sources':['https://www.fao.org/land-water/resources/tools/databases/ecocrop/en',
                        'https://www.nrcs.usda.gov/sites/default/files/2024-11/NRCS-FactSheet-Conservation%20Crop%20Rotation%20%E2%80%93%20Climate-Smart%20Plant%20Cover.pdf']}


def score(metrics, soil_priority):
    # Índice comparativo transparente, NOT agronomic probability.
    w=soil_priority/100
    resilience=max(0.,100-metrics['stress_pct'])
    return round((1-w)*resilience+w*metrics['soil_proxy'],1)

def evaluate(rot, per_crop, req):
    # Todos los cultivos se evalúan una sola vez por solicitud; evita miles de simulaciones duplicadas.
    import numpy as np
    from science_engine import CROPS
    metrics=[per_crop[c] for c in rot]
    coverage=float(np.mean([CROPS[c]['cover'] for c in rot]))
    diversity=len(set(rot))/3
    legume=sum(CROPS[c]['legume'] for c in rot)/3
    return {'soil_proxy':100*(.45*coverage+.30*diversity+.25*legume),
            'stress_pct':float(np.mean([m['stress_pct'] for m in metrics])),
            'growth_proxy':float(np.mean([m['growth_proxy'] for m in metrics]))}

def distinct_option(rows, selected, metric, pred=lambda x:True):
    candidates=[x for x in rows if x['rotation'] not in selected and pred(x)]
    return max(candidates,key=lambda v:v[metric]) if candidates else None

@app.post('/api/recommend')
def recommend(req:Request):
    try:
        if req.start>req.end: raise ValueError('El año inicial debe ser anterior o igual al final.')
        if req.current_crop not in CROPS: raise ValueError('Cultivo actual no reconocido.')
        choices=list(dict.fromkeys(req.available_crops))
        if len(choices)<2 or any(c not in CROPS for c in choices):
            raise ValueError('Selecciona al menos dos cultivos válidos.')
        if req.current_crop not in choices: choices.insert(0,req.current_crop)
        df=climate(round(req.lat,3),round(req.lon,3),req.start,req.end,req.source).copy()
        windows=season_windows(df,req.sow_month)
        if not windows:raise ValueError('No hay una temporada completa. Prueba otro mes o un período más amplio.')
        win=windows[-1]
        # La textura modifica la capacidad efectiva de agua, de forma orientativa.
        # Sin análisis de profundidad, densidad y retención no es una medición local.
        texture_factor={'Arenoso':0.65,'Franco':1.0,'Arcilloso':0.85,'Desconocido':1.0}[req.soil_texture]
        effective_capacity=req.soil_capacity_mm*texture_factor
        per_crop={c:crop_season(c,win,req.lat,req.irrigation_mm,req.rain_reduction_pct,
                       req.warming_c,effective_capacity)[1] for c in choices}
        # Escenario de referencia: mismos cultivos, clima y riego, sin perturbación.
        # No inventar una penalización cuando el motor no detecta estrés adicional.
        if req.warming_c or req.rain_reduction_pct:
            reference_crop={c:crop_season(c,win,req.lat,req.irrigation_mm,0,0,
                             effective_capacity)[1] for c in choices}
        else:
            reference_crop=per_crop
        baseline=(req.current_crop,)*3
        rows=[]
        for rot in product(choices,repeat=3):
            m=evaluate(rot,per_crop,req)
            soil_eval=evaluate_soil(rot,req.soil_texture,req.drainage,req.salinity,req.organic_matter)
            rows.append({'rotation':rot,'soil':round(m['soil_proxy'],1),
                         'stress':round(m['stress_pct'],1),'growth':round(m['growth_proxy'],1),
                         'rating':round(max(0,min(100,score(m,req.soil_priority)+soil_eval['rotation_adjustment'])),1),
                         'soil_eval':soil_eval})
        byrot={r['rotation']:r for r in rows}
        base=byrot[baseline]
        selected=[baseline]
        diversified=distinct_option(rows,selected,'soil',lambda x:len(set(x['rotation']))>=2)
        if diversified:selected.append(diversified['rotation'])
        hardy=distinct_option(rows,selected,'rating',lambda x:len(set(x['rotation']))>=2)
        if hardy:selected.append(hardy['rotation'])
        # Fallback: 2 choices -> 8 combos, still 3 rotations.
        if len(selected)<3:
            other=distinct_option(rows,selected,'rating')
            if other:selected.append(other['rotation'])
        # Mostrar además una rotación que incluya cada cultivo marcado.
        # Las primeras tres tarjetas conservan el esquema de comparación original.
        # Cada tarjeta adicional cubre el mayor número posible de especies pendientes.
        missing=set(choices)-set().union(*(set(r) for r in selected))
        while missing:
            candidates=[r for r in rows if r['rotation'] not in selected and set(r['rotation']) & missing]
            if not candidates: break
            extra=max(candidates,key=lambda r:(len(set(r['rotation']) & missing),r['rating']))
            selected.append(extra['rotation'])
            missing.difference_update(extra['rotation'])
        labels=['Tu práctica actual','Alternativa para el suelo','Alternativa equilibrada']
        labels.extend(f'Alternativa con cultivos seleccionados {i}' for i in range(1,max(0,len(selected)-3)+1))
        explanations=[]
        for i,rot in enumerate(selected):
            r=byrot[rot]
            diversity=len(set(rot))
            legume=any(CROPS[c]['legume'] for c in rot)
            if i==0:
                main='Esta es la referencia para comparar; repetir un cultivo puede mantener riesgos de plagas y agotamiento de nutrientes.'
            elif i==1:
                main='Prioriza una mayor diversidad y cobertura teórica del suelo en este modelo.'
            elif i==2:
                main='Busca equilibrar el indicador de estrés climático con el de protección del suelo.'
            else:
                main='Esta alternativa permite visualizar cultivos que seleccionaste y que no aparecían en las tres primeras propuestas. No implica que sea agronómicamente viable.'
            notes=[]
            if diversity>1: notes.append('Alterna especies y puede interrumpir ciclos de plagas.')
            if legume: notes.append('Incluye una leguminosa que podría contribuir nitrógeno si se maneja adecuadamente.')
            if not legume: notes.append('No incorpora una leguminosa; conviene revisar el manejo de nutrientes.')
            notes.extend(r['soil_eval']['warnings'])
            notes.append('Confirma la compatibilidad de cada cultivo con la temporada, el mercado y el agua disponible.')
            reference_metrics=evaluate(rot,reference_crop,req)
            reference_score=round(max(0,min(100,score(reference_metrics,req.soil_priority)+r['soil_eval']['rotation_adjustment'])),1)
            reference_stress=round(reference_metrics['stress_pct'],1)
            explanations.append({'reference_score':reference_score,
                 'score_change':round(r['rating']-reference_score,1),
                 'reference_stress_pct':reference_stress,
                 'stress_change_pct':round(r['stress']-reference_stress,1),
                 'id':f'option{i}','label':labels[i], 'rotation':list(rot),
                 'score':r['rating'],'soil_score':r['soil'],'stress_pct':r['stress'],
                 'growth_index':r['growth'],'explanation':main,'notes':notes,
                 'suggested_actions':r['soil_eval']['actions'],'crop_groups':r['soil_eval']['family_groups']})
        winner=max(explanations,key=lambda x:x['score'])
        delta=round(winner['score']-explanations[0]['score'],1)
        simple=(f'Para la zona y condiciones seleccionadas, la combinación {" → ".join(winner["rotation"])} '
                f'obtuvo el índice comparativo más alto ({winner["score"]}/100). '
                f'La diferencia respecto de la práctica actual fue de {delta:+.1f} puntos. '
                'Esto NO significa que vaya a producir más cosecha: compara únicamente estrés climático y '
                'un indicador teórico de protección del suelo. Antes de cambiar la siembra, revisa suelo, '
                'disponibilidad real de riego y asesoría agrícola local.')
        rain=float(df.PRECTOTCORR.mean()*365.25)
        temp=float(df.T2M.mean())
        return {'options':explanations,'summary':simple,
                'scenario':{'warming_c':req.warming_c,'rain_reduction_pct':req.rain_reduction_pct,'soil_priority_pct':req.soil_priority,
                            'message':('Comparación contra el mismo clima histórico sin calentamiento ni reducción de lluvias. Los cambios pueden ser nulos si el riego compensa la sequía, el estrés ya es máximo o la puntuación prioriza solo el suelo.')},
                'climate':{'mean_temp_c':round(temp,1),'mean_rain_mm_year':round(rain,1),
                           'season_year':int(win.date.iloc[0].year)},
                'soil_model':{'capacity_mm_user':req.soil_capacity_mm,'capacity_mm_effective':round(effective_capacity,1),'texture_factor':texture_factor,'notice':'Factor de textura orientativo, no medición de retención hídrica'},
                'soil_inputs':{'texture':req.soil_texture,'drainage':req.drainage,'salinity':req.salinity,
                               'organic_matter':req.organic_matter},
                'considered_crops':choices,'displayed_crops':sorted(set().union(*(set(r) for r in selected))),
                'score_method':{'resilience_weight_pct':100-req.soil_priority,'soil_weight_pct':req.soil_priority,'formula':'Índice = (1 - prioridad_suelo/100) × (100 - estrés_medio) + (prioridad_suelo/100) × indicador_suelo + ajuste_heurístico_por_familias; limitado entre 0 y 100','meaning':'Índice relativo experimental, no probabilidad de éxito ni rendimiento esperado'},
                'source':'NASA POWER, datos históricos' if req.source=='nasa' else 'DATOS SINTÉTICOS DE DEMOSTRACIÓN',
                'limitations':['Los grupos botánicos y beneficios potenciales se apoyan en FAO/USDA; los coeficientes numéricos de simulación son supuestos exploratorios no calibrados.',
                  'Cada ciclo se evalúa bajo una temporada histórica de referencia; no modela acumulación real de nutrientes entre años.',
                  'La textura ajusta orientativamente la capacidad hídrica. Drenaje y salinidad producen alertas, no penalizaciones numéricas sin medidas de campo.',
                  'El puntaje incluye un ajuste heurístico pequeño por diversidad de grupos, no un valor medido de fertilidad.',
                  'Los índices no son probabilidades de supervivencia, pronósticos de rendimiento ni mediciones reales de fertilidad.',
                  'Si usas datos de demostración, las condiciones climáticas no son observaciones reales de la zona.']}
    except Exception as e:
        raise HTTPException(status_code=422,detail=str(e)) from e
