"""Agroguard v3: exploratory daily crop water balance & thermal development.
Not calibrated for crop yields, survival, soil nutrition, or space cultivation.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# Placeholder agronomic parameters; illustrative and MUST be locally calibrated.
# kc: generic initial, midseason, late-season stage crop coefficients.
CROPS = {
    'Trigo': {'base': 4, 'gdd': 1350, 'heat': 32, 'kc': (.40, 1.12, .42), 'cover': .65, 'legume': False},
    'Cebada': {'base': 3, 'gdd': 1200, 'heat': 32, 'kc': (.38, 1.05, .40), 'cover': .65, 'legume': False},
    'Quinua': {'base': 5, 'gdd': 1250, 'heat': 34, 'kc': (.40, 1.00, .48), 'cover': .52, 'legume': False},
    'Leguminosa (genérica)': {'base': 5, 'gdd': 1150, 'heat': 31, 'kc': (.40, 1.08, .45), 'cover': .75, 'legume': True},
}

# NUEVOS CULTIVOS: grupos y funciones agronómicas generales respaldados por FAO/USDA.
# Los valores base/gdd/heat/kc/cover siguientes son HIPÓTESIS de simulación, NO datos medidos ni tablas oficiales.
CROPS.update({
    'Avena': {"base": 3, "gdd": 1250, "heat": 32, "kc": (0.4, 1.05, 0.45), "cover": 0.7, "legume": False},
    'Centeno': {"base": 2, "gdd": 1200, "heat": 31, "kc": (0.4, 1.07, 0.42), "cover": 0.82, "legume": False},
    'Maíz': {"base": 8, "gdd": 1500, "heat": 35, "kc": (0.45, 1.17, 0.65), "cover": 0.55, "legume": False},
    'Sorgo': {"base": 10, "gdd": 1450, "heat": 38, "kc": (0.38, 1.05, 0.48), "cover": 0.6, "legume": False},
    'Lenteja': {"base": 4, "gdd": 1100, "heat": 32, "kc": (0.38, 1.0, 0.42), "cover": 0.62, "legume": True},
    'Garbanzo': {"base": 5, "gdd": 1250, "heat": 35, "kc": (0.38, 1.0, 0.42), "cover": 0.58, "legume": True},
    'Arveja': {"base": 4, "gdd": 1000, "heat": 30, "kc": (0.4, 1.07, 0.45), "cover": 0.65, "legume": True},
    'Poroto': {"base": 8, "gdd": 1150, "heat": 34, "kc": (0.4, 1.1, 0.45), "cover": 0.6, "legume": True},
    'Girasol': {"base": 6, "gdd": 1300, "heat": 36, "kc": (0.35, 1.1, 0.45), "cover": 0.55, "legume": False},
    'Canola': {"base": 4, "gdd": 1250, "heat": 33, "kc": (0.4, 1.1, 0.4), "cover": 0.65, "legume": False},
    'Papa': {"base": 5, "gdd": 1100, "heat": 31, "kc": (0.45, 1.12, 0.75), "cover": 0.45, "legume": False},
    'Rábano forrajero': {"base": 4, "gdd": 1000, "heat": 32, "kc": (0.4, 1.05, 0.42), "cover": 0.8, "legume": False},
    'Vicia': {"base": 4, "gdd": 1050, "heat": 31, "kc": (0.4, 1.03, 0.43), "cover": 0.85, "legume": True},
})

ROTATIONS = {
    'Actual · trigo continuo': ('Trigo', 'Trigo', 'Trigo'),
    'Diversificada': ('Trigo', 'Leguminosa (genérica)', 'Cebada'),
    'Ahorro hídrico · candidata': ('Quinua', 'Leguminosa (genérica)', 'Cebada'),
}


def extraterrestrial_radiation(latitude: float, doy: int) -> float:
    """FAO-56 Ra in MJ m^-2 day^-1, for latitude and day of year."""
    phi = math.radians(latitude)
    dr = 1 + .033 * math.cos(2 * math.pi * doy / 365)
    solar_decl = .409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    x = -math.tan(phi) * math.tan(solar_decl)
    sunset = math.acos(max(-1.0, min(1.0, x)))
    return max(0., 24 * 60 / math.pi * .0820 * dr * (
        sunset * math.sin(phi) * math.sin(solar_decl) +
        math.cos(phi) * math.cos(solar_decl) * math.sin(sunset)))


def hargreaves_et0(tmean: float, tmin: float, tmax: float, latitude: float, doy: int) -> float:
    """Hargreaves-Samani reference ET0, mm/day; approximate; no humidity/wind correction."""
    ra_mm = .408 * extraterrestrial_radiation(latitude, doy)
    return max(0., .0023 * max(0., tmean + 17.8) * math.sqrt(max(0., tmax-tmin)) * ra_mm)


def season_windows(df: pd.DataFrame, start_month: int, length: int = 120):
    """Return complete date-contiguous windows beginning on 1st of selected month."""
    t = df.copy().sort_values('date').drop_duplicates('date').reset_index(drop=True)
    t['date'] = pd.to_datetime(t['date'])
    windows = []
    for year in sorted(t['date'].dt.year.unique()):
        day = pd.Timestamp(year=int(year), month=int(start_month), day=1)
        dates = pd.date_range(day, periods=length, freq='D')
        sel = t[(t['date'] >= dates[0]) & (t['date'] <= dates[-1])]
        if len(sel) == length and sel['date'].reset_index(drop=True).equals(pd.Series(dates, name='date')):
            if sel[['T2M', 'T2M_MAX', 'T2M_MIN', 'PRECTOTCORR']].notna().all().all():
                windows.append(sel.reset_index(drop=True))
    return windows


def crop_season(crop_name: str, weather: pd.DataFrame, latitude: float = -22.91,
                irrigation_mm: float = 350, rain_reduction: float = 0, warming: float = 0,
                soil_capacity_mm: float = 120, initial_fraction: float = .55,
                uncertainty_temp: float = 0, rain_multiplier: float = 1.0):
    """Simple root zone 'bucket' + ET0-Hargreaves + GDD development proxies.
    irrigation_mm is season total, distributed uniformly (not an irrigation schedule).
    """
    if crop_name not in CROPS: raise ValueError('Cultivo desconocido')
    if len(weather) < 20: raise ValueError('Clima insuficiente')
    if not (0 <= rain_reduction <= 100 and 0 <= initial_fraction <= 1):
        raise ValueError('Parámetros fuera de rango')
    crop = CROPS[crop_name]
    water = float(soil_capacity_mm) * initial_fraction
    gdd_total = 0.
    cumulative_actual = 0.
    cumulative_potential = 0.
    heat_days = 0
    rain_total = 0.
    applied_irrigation = 0.
    out = []
    for i, row in enumerate(weather.itertuples(index=False)):
        date = pd.Timestamp(row.date)
        temp = float(row.T2M) + warming + uncertainty_temp
        tmax = float(row.T2M_MAX) + warming + uncertainty_temp
        tmin = float(row.T2M_MIN) + warming + uncertainty_temp
        if tmax < tmin: tmax, tmin = tmin, tmax
        rain = max(0., float(row.PRECTOTCORR) * (1 - rain_reduction/100) * rain_multiplier)
        gdd_total += max(0., temp - crop['base'])
        development = min(1., gdd_total / crop['gdd'])
        if development < .2: stage = 'Establecimiento'; kc = crop['kc'][0]
        elif development < .7: stage = 'Desarrollo'; kc = crop['kc'][1]
        else: stage = 'Maduración'; kc = crop['kc'][2]
        et0 = hargreaves_et0(temp, tmin, tmax, latitude, date.dayofyear)
        etc = et0 * kc
        irrig = irrigation_mm / len(weather)
        # 80% rain effectiveness, 90% irrigation efficiency: illustrative assumptions.
        water = min(soil_capacity_mm, water + rain*.80 + irrig*.90)
        supplied = min(water, etc)
        water -= supplied
        water_factor = 1. if etc < 1e-8 else supplied / etc
        heat_factor = max(0., 1 - max(0., tmax - crop['heat'])/14)
        response = water_factor * heat_factor
        cumulative_actual += max(0., temp-crop['base']) * response
        cumulative_potential += max(0., temp-crop['base'])
        heat_days += int(tmax > crop['heat'])
        rain_total += rain
        applied_irrigation += irrig
        out.append({'date':date, 'day':i+1,'stage':stage, 'development_pct':100*development,
                    'temperature_c':temp,'rain_mm':rain,'irrigation_mm':irrig, 'et0_mm':et0,
                    'etc_mm':etc,'soil_water_mm':water,'water_factor':water_factor,
                    'heat_factor':heat_factor,'growth_proxy_pct':100*min(1.,cumulative_actual/crop['gdd']),
                    'stress_proxy_pct':100*(1-response)})
    trace = pd.DataFrame(out)
    growth_proxy = float(trace['growth_proxy_pct'].iloc[-1])
    stress_pct = float(trace['stress_proxy_pct'].mean())
    completion = float(trace['development_pct'].iloc[-1])
    return trace, {'growth_proxy':growth_proxy,'stress_pct':stress_pct,
                   'completion_pct':completion,'heat_days':heat_days,
                   'rain_mm':rain_total,'irrigation_mm':applied_irrigation,
                   'water_final_mm':water,'avg_et0_mm':float(trace['et0_mm'].mean())}


def rotation_run(rotation, windows, latitude, irrigation_mm, rain_reduction, warming,
                 capacity_mm, rng):
    """Compare all rotations against a shared climate window within each scenario."""
    season_metrics = []
    for crop, window in zip(rotation, windows):
        trace, m = crop_season(crop, window, latitude, irrigation_mm,
                               rain_reduction,warming,capacity_mm,
                               uncertainty_temp=float(rng['temp']),rain_multiplier=float(rng['rain']))
        season_metrics.append(m)
    # Cover/diversity/legume are placeholders, not measured soil outcomes.
    diversity = len(set(rotation))/3
    legume = sum(CROPS[c]['legume'] for c in rotation)/3
    coverage = float(np.mean([CROPS[c]['cover'] for c in rotation]))
    soil_proxy = 100*(.45*coverage+.30*diversity+.25*legume)
    growth = float(np.mean([x['growth_proxy'] for x in season_metrics]))
    stress = float(np.mean([x['stress_pct'] for x in season_metrics]))
    return {'growth_proxy':growth, 'stress_pct':stress, 'soil_proxy':soil_proxy,
            'score':.75*growth+.25*soil_proxy,
            'irrigation_mm':3*irrigation_mm}


def scenario_generator(df, latitude, irrigation_mm, rain_reduction, warming,
                       capacity_mm, start_month, n=300, seed=2026):
    """Yields matched rotation comparisons, one scenario at a time."""
    if not (1 <= n <= 5000): raise ValueError('Número de escenarios fuera de rango')
    windows = season_windows(df, start_month)
    if not windows: raise ValueError('No hay temporadas completas para el mes elegido. Cambia el mes o los años.')
    rng = np.random.default_rng(seed)
    for i in range(n):
        selected = [windows[j] for j in rng.integers(0,len(windows),size=3)]
        perturb = {'temp':float(rng.normal(0,0.6)), 'rain':float(rng.lognormal(-.5*.12**2,.12))}
        row = {'scenario':i+1}
        for name, rot in ROTATIONS.items():
            m = rotation_run(rot, selected, latitude, irrigation_mm, rain_reduction,
                             warming, capacity_mm, perturb)
            row[name]=round(m['score'],2)
            row[name+' | crecimiento']=round(m['growth_proxy'],2)
            row[name+' | estrés']=round(m['stress_pct'],2)
        yield row


def summarize_results(results: pd.DataFrame, source_label: str, latitude: float, longitude: float,
                      params: dict) -> str:
    """Plain-language rule-based narrative; no model/API/AI service required."""
    names = list(ROTATIONS)
    med = results[names].median().sort_values(ascending=False)
    best, worst = med.index[0], med.index[-1]
    gap = float(med.iloc[0] - med.iloc[-1])
    stress_col = best + ' | estrés'
    stress = float(results[stress_col].median())
    p10,p90 = results[best].quantile([.1,.9])
    if gap < 3:
        comparison = 'Las diferencias entre las estrategias son pequeñas en esta simulación; no hay una ganadora clara.'
    else:
        comparison = f'La estrategia «{best}» obtuvo el mayor índice comparativo; superó a «{worst}» por {gap:.1f} puntos de índice en la mediana.'
    if stress > 60: stress_meaning = 'Los cultivos simulados enfrentaron estrés frecuente o intenso en esta configuración.'
    elif stress > 30: stress_meaning = 'Los cultivos simularon períodos de estrés apreciable, aunque no permanente.'
    else: stress_meaning = 'El estrés climático e hídrico promedio del modelo fue relativamente bajo.'
    ranked = '\n'.join(f'- {i+1}. {name}: {value:.1f}/100 (índice exploratorio)' for i,(name,value) in enumerate(med.items()))
    return f'''# Agroguard — Informe comprensible de la simulación

## ¿Qué estudiamos?
Comparamos **tres secuencias de cultivos** bajo condiciones climáticas basadas en **{source_label}**, para las coordenadas {latitude:.3f}, {longitude:.3f}. Ejecutamos **{len(results)} escenarios simulados** con un aumento de **{params['warming']} °C**, reducción de lluvia del **{params['rain_reduction']} %** y **{params['irrigation_mm']} mm de riego por temporada**.

## ¿Qué encontramos?
{comparison}

{ranked}

El resultado de la mejor estrategia varió entre **{p10:.1f} y {p90:.1f} puntos** en el 80 % central aproximado de escenarios. Esta variación describe el comportamiento **del simulador**, no la incertidumbre real de cosechas.

## ¿Qué significa en palabras simples?
Cada cultivo necesita agua y temperaturas adecuadas. El software calcula diariamente cuánta agua podría quedar en el suelo y cómo esas condiciones podrían limitar el desarrollo. Después reúne esa información con indicadores simplificados de diversidad y cobertura vegetal para **comparar alternativas**, no para asegurar una cosecha.

{stress_meaning} La estrategia mejor clasificada mostró un **estrés mediano de {stress:.1f} %** según el indicador interno del modelo.

## ¿Qué deberíamos hacer con este resultado?
La alternativa mejor clasificada es una **hipótesis para investigar primero**. Antes de sembrar, comprobar disponibilidad real de riego, salinidad, tipo de suelo, clima local, plagas y viabilidad comercial con especialistas y agricultores.

## Lo que todavía NO sabemos
- No estamos prediciendo kilos cosechados ni probabilidades reales de supervivencia.
- Los coeficientes de cultivos, las temporadas y el indicador de suelo **no están calibrados** con experimentos de Atacama.
- Los datos satelitales representan condiciones meteorológicas de una grilla, no una medición directa del suelo de la parcela.
- Una buena puntuación **no demuestra** que el cultivo sea viable en Atacama ni en un hábitat espacial.

**Conclusión:** Agroguard permite descartar o priorizar ideas de rotación para futuras pruebas. El siguiente paso científico es contrastar estas predicciones exploratorias con ensayos controlados.
'''
