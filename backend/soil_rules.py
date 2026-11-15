"""Reglas heurísticas explícitas. NO son valores agronómicos calibrados."""
from collections import Counter

TEXTURES = ['Arenoso', 'Franco', 'Arcilloso', 'Desconocido']
DRAINAGES = ['Bueno', 'Moderado', 'Deficiente', 'Desconocido']
SALINITIES = ['Baja', 'Media', 'Alta', 'Desconocida']
ORGANIC = ['Baja', 'Media', 'Alta', 'Desconocida']
# Familias funcionales simplificadas; 'Leguminosa (genérica)' no equivale a una variedad.
FAMILIES = {'Trigo':'Cereal','Cebada':'Cereal','Quinua':'Amarantácea','Leguminosa (genérica)':'Leguminosa'}
FAMILIES.update({'Avena': 'Cereal', 'Centeno': 'Cereal', 'Maíz': 'Cereal', 'Sorgo': 'Cereal', 'Lenteja': 'Leguminosa', 'Garbanzo': 'Leguminosa', 'Arveja': 'Leguminosa', 'Poroto': 'Leguminosa', 'Girasol': 'Asterácea', 'Canola': 'Brassicácea', 'Papa': 'Solanácea', 'Rábano forrajero': 'Brassicácea', 'Vicia': 'Leguminosa'})

def evaluate_soil(rotation, texture, drainage, salinity, organic):
    """Devuelve señales orientativas y restricciones, nunca rendimientos medidos."""
    warnings=[]; recommendations=[]
    if texture == 'Arenoso':
        warnings.append('El suelo arenoso puede almacenar menos agua aprovechable: comprueba riego y profundidad de raíces.')
        recommendations.append('Considerar cobertura vegetal o residuos para disminuir evaporación y erosión.')
    elif texture == 'Arcilloso':
        warnings.append('En suelos arcillosos, verificar infiltración y compactación antes de definir el riego.')
    elif texture == 'Desconocido':
        warnings.append('No se conoce la textura del suelo; una prueba de campo mejoraría la evaluación hídrica.')
    if drainage == 'Deficiente':
        warnings.append('El drenaje deficiente puede favorecer encharcamiento y problemas radiculares.')
        recommendations.append('Revisar el drenaje del terreno con asistencia local antes de sembrar.')
    elif drainage == 'Desconocido':
        warnings.append('Drenaje sin verificar; puede modificar la viabilidad de la rotación.')
    if salinity == 'Alta':
        warnings.append('Salinidad alta declarada: es necesario medir conductividad eléctrica y revisar tolerancia de cada variedad.')
        recommendations.append('Solicitar análisis de salinidad de suelo y agua de riego antes de elegir cultivos.')
    elif salinity == 'Media':
        warnings.append('Salinidad media declarada: comprobar tolerancia de cultivos y calidad del agua.')
    elif salinity == 'Desconocida':
        warnings.append('Salinidad desconocida; no podemos validar compatibilidad del cultivo.')
    if organic == 'Baja':
        recommendations.append('Explorar cultivo de cobertura, manejo de residuos y aporte de materia orgánica según asesoría local.')
    if organic == 'Desconocida':
        warnings.append('Materia orgánica no medida: la estimación de salud del suelo es muy limitada.')
    families=[FAMILIES[c] for c in rotation]
    repeats=sum(families[i]==families[i-1] for i in range(1,len(families)))
    if repeats:
        warnings.append('La rotación repite grupos de cultivos en ciclos consecutivos; revisar plagas, enfermedades y nutrientes.')
    if len(set(rotation))>1:
        recommendations.append('La diversificación puede ayudar a interrumpir ciclos de plagas, según especies y manejo.')
    if 'Leguminosa (genérica)' in rotation:
        recommendations.append('Una leguminosa compatible podría fijar nitrógeno si cuenta con las condiciones y microorganismos adecuados.')
    # Ajuste pequeño y explícito sólo para ORDENAR alternativas: no estimación de fertilidad.
    bonus= min(8.0, len(set(families))*2.0) - repeats*3.0
    if organic == 'Baja' and len(set(rotation))>1: bonus+=2.0
    # La salinidad es un aviso de viabilidad, NO podemos ordenar variedades sin datos.
    return {'rotation_adjustment':round(bonus,1),'warnings':list(dict.fromkeys(warnings)),
            'actions':list(dict.fromkeys(recommendations)),'family_groups':families}
