from pathlib import Path
import json
import pandas as pd


EXPECTED = {'MRCLM': (30, 752, 51830), 'OMITSJU': (10, 139, 12176),
            'CITSV': (15, 251, 16046), 'MRCLASSO': (13, 521, 39000),
            'LRRS': (34, 520, 34078), 'CITVR': (13, 283, 30791),
            'CITCRC': (15, 270, 17588), 'CITLA': (43, 1344, 120287),
            'CITSO': (16, 372, 30572), 'CITPI': (28, 639, 14053), 'CITHSL': (2, 107, 5340)}


def audit_csv(path, output):
                                        
    d = pd.read_csv(path, dtype=str)
    d['boarding'] = pd.to_numeric(d.montant, errors='coerce')
    d['scheduled'] = pd.to_datetime(d.date_heure_depart_prevue, errors='coerce')
    rows = []
    for region, x in d.groupby('organisation'):
        actual = [int(x.ligne.nunique()), int(x.arret_code.nunique()), float(x.boarding.sum())]
        expected = EXPECTED.get(region)
        rows.append(dict(region=region, actual_routes_stops_boardings=actual,
            case_study_routes_stops_boardings=expected,
            matches_table=actual == list(expected) if expected else False,
            first=str(x.scheduled.min()), last=str(x.scheduled.max()), records=len(x)))
    result = dict(records=len(d), regions=rows,
                  note='Raw-file statistics; study-specific exclusions or GTFS inventory may change counts')
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
