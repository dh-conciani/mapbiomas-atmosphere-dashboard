"""Build a static, publishable MapBiomas Atmosphere dashboard dataset.

Inputs: REAL XLSX v02 workbooks from GCS and REAL Earth Engine territory geometries.
No demonstration data, no Earth Engine zonal-statistics recomputation.
Run in authenticated Colab; the deployed Streamlit app needs no GCP credentials.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT = 'mapbiomas-brazil'
BUCKET = 'mapbiomas-fire'
XLSX_PREFIX = 'temp/ATMOSPHERE/stats/v01/merged/xlsx_en_atbd_v02/'
GEO_PREFIX = 'temp/ATMOSPHERE/stats/v01/merged/dashboard_geometries_v01/'
TERRITORIES = {
    'brazil': 'projects/mapbiomas-territories/assets/PLATFORM/demo/mapbiomas/brazil/territories/BRASIL_IBGE_2025_D71F57FE/cc8a1562-88be-4106-8701-34da9e3a3a40',
    'regions': 'projects/mapbiomas-territories/assets/PLATFORM/demo/mapbiomas/brazil/territories/REGIOES_IBGE_2025_2CDBB85D/97f303cb-f543-400c-9c8f-47cc6b9b3a66',
    'states': 'projects/mapbiomas-territories/assets/PLATFORM/demo/mapbiomas/brazil/territories/ESTADOS_IBGE_2025_F7EAF953/3f95fd9d-4b26-4243-8612-37b9d5b12a55',
    'biomes': 'projects/mapbiomas-territories/assets/TERRITORIES/BRAZIL/WORKSPACE/BIOMES/BIOMES_v4',
    'legal_amazon': 'projects/mapbiomas-territories/assets/TERRITORIES-STAGING/BRAZIL/WORKSPACE/LEGAL_AMAZON/LEGAL_AMAZON_v3',
    'matopiba': 'projects/mapbiomas-territories/assets/TERRITORIES-STAGING/BRAZIL/WORKSPACE/MATOPIBA/MATOPIBA_v3',
    'hidrographic_regions': 'projects/mapbiomas-territories/assets/TERRITORIES/BRAZIL/WORKSPACE/HYDROGRAPHIC_REGIONS/HYDROGRAPHIC_REGIONS_v1',
    'indigenous_lands': 'projects/mapbiomas-territories/assets/PLATFORM/demo/mapbiomas/brazil/territories/TERRAS_INDIGENAS_FUNAI_2026_F7F13F79/8b94c232-c551-4dd1-ac0c-2b40145db021',
}
LABELS = {
    'brazil': 'Brazil', 'regions': 'Regions', 'states': 'States', 'biomes': 'Biomes',
    'legal_amazon': 'Legal Amazon', 'matopiba': 'MATOPIBA',
    'hidrographic_regions': 'Hydrographic regions', 'indigenous_lands': 'Indigenous lands',
}
EXPECTED_SHEETS = ('Data_Dictionary', 'Annual', 'Monthly', 'Multiyear', 'Territories')
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
R_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
P_NS = '{http://schemas.openxmlformats.org/package/2006/relationships}'


def workbook_name(level: str):
    return f'atmosphere_{level}_1985_2025_v02.xlsx'


def stream_xlsx_sheet(xlsx_path, sheet_name):
    """Streaming XLSX reader; works with long Monthly sheets, preserves text IDs.

    Native zip/xml implementation avoids loading whole XLSX worksheets into RAM.
    """
    with zipfile.ZipFile(xlsx_path) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        relation = {x.get('Id'): x.get('Target') for x in rels.findall(f'{P_NS}Relationship')}
        sheets = {x.get('name'): x.get(f'{R_NS}id') for x in wb.findall(f'{NS}sheets/{NS}sheet')}
        if sheet_name not in sheets:
            raise KeyError(f'{sheet_name} not found in {xlsx_path}; found {list(sheets)}')
        target = relation[sheets[sheet_name]]
        xml_path = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        if xml_path not in z.namelist() and target.startswith('xl/'):
            xml_path = target
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings = [''.join(t.text or '' for t in si.iter(f'{NS}t')) for si in root.findall(f'{NS}si')]
        with z.open(xml_path) as xml:
            context = ET.iterparse(xml, events=('start', 'end'))
            _, root = next(context)
            sheet_data_elem = None
            for event, elem in context:
                if event == 'start' and elem.tag == f'{NS}sheetData':
                    sheet_data_elem = elem
                if event != 'end' or elem.tag != f'{NS}row':
                    continue
                row = []
                for cell in elem.findall(f'{NS}c'):
                    cell_address = cell.get('r', 'A1')
                    letter = re.match(r'[A-Z]+', cell_address).group()
                    index = 0
                    for ch in letter:
                        index = index * 26 + ord(ch) - 64
                    index -= 1
                    if index >= 2000:
                        raise ValueError('Unexpectedly wide XLSX: ' + cell_address)
                    while len(row) <= index:
                        row.append(None)
                    typ = cell.get('t')
                    node = cell.find(f'{NS}v')
                    raw = node.text if node is not None else None
                    if typ == 's' and raw is not None:
                        value = strings[int(raw)]
                    elif typ == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter(f'{NS}t'))
                    elif typ in ('str', 'e'):
                        value = raw
                    elif typ == 'b' and raw is not None:
                        value = raw == '1'
                    elif raw is None or raw == '':
                        value = None
                    else:
                        try:
                            as_float = float(raw)
                            value = int(as_float) if as_float.is_integer() and abs(as_float) <= 2**53 else as_float
                        except (TypeError, ValueError):
                            value = raw
                    row[index] = value
                yield row
                elem.clear()
                # Clear the current sheetData children, not just the XML root.
                # Otherwise a million-row Indigenous Lands worksheet leaks empty row elements.
                if sheet_data_elem is not None:
                    sheet_data_elem.clear()


def _sheet_data(xlsx, name):
    it = iter(stream_xlsx_sheet(xlsx, name))
    header = [str(c).strip() if c is not None else '' for c in next(it)]
    if len(header) != len(set(header)):
        raise ValueError(f'Duplicate worksheet columns in {name}: {header}')
    return header, it


def _to_clean_str(value):
    return '' if value is None else str(value).strip()


def _to_float(value):
    if value is None or value == '':
        return None
    return float(value)


def _to_int(value):
    if value is None or value == '':
        return None
    return int(value)


def download_xlsx_from_gcs(workdir='.', allow_missing_indigenous=True, force=False):
    """Download REAL v02 XLSX exports from an authenticated GCS bucket."""
    from google.cloud import storage
    workdir = Path(workdir)
    dest = workdir / 'source_xlsx'
    dest.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    result = {}
    for territory in TERRITORIES:
        name = workbook_name(territory)
        blob = bucket.blob(XLSX_PREFIX + name)
        target = dest / name
        if not blob.exists(client=client):
            if territory == 'indigenous_lands' and allow_missing_indigenous:
                print(f'NOT READY: {territory} XLSX missing; rerun after it finishes.')
                continue
            raise FileNotFoundError(f'gs://{BUCKET}/{blob.name}')
        blob.reload(client=client)
        if force or not target.exists() or target.stat().st_size != blob.size:
            blob.download_to_filename(str(target))
        result[territory] = target
        print(f'XLSX READY: {territory:23} {target.stat().st_size / 1e6:.1f} MB')
    if not result:
        raise ValueError('No v02 workbooks found. Run XLSX v02 generation first.')
    return result


def _catalog_from_xlsx(path):
    header, it = _sheet_data(path, 'Data_Dictionary')
    records = []
    for raw in it:
        vals = dict(zip(header, raw))
        col = _to_clean_str(vals.get('column_name'))
        if not col:
            continue
        p = _to_clean_str(vals.get('period'))
        if p not in ('Annual', 'Monthly', 'Multiyear'):
            continue
        records.append({k: _to_clean_str(vals.get(k)) for k in header})
    return records


def _write_territories(path, out_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    header, data = _sheet_data(path, 'Territories')
    if 'territory_id' not in header or 'territory_name' not in header:
        raise ValueError('Territories sheet needs territory_id and territory_name')
    rows = []
    for raw in data:
        values = dict(zip(header, raw))
        tid = _to_clean_str(values.get('territory_id'))
        if not tid:
            continue
        name = _to_clean_str(values.get('territory_name'))
        item = {h: _to_clean_str(values.get(h)) for h in header}
        item['territory_id'] = tid
        item['territory_name'] = name
        rows.append(item)
    if len(rows) != len({x['territory_id'] for x in rows}):
        raise ValueError(f'Duplicated territory_id in Territories sheet: {path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path, compression='zstd')
    else:
        raise ValueError(f'No territory records in {path}')
    return len(rows)


def _build_period(path, level, period, output_dir, batch_size=800):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import collections
    names = [period]
    # XLSX v02 may split very long sheets into Monthly_02, Monthly_03, ...
    with zipfile.ZipFile(path) as z:
        book = ET.fromstring(z.read('xl/workbook.xml'))
        all_sheets = [x.get('name') for x in book.findall(f'{NS}sheets/{NS}sheet')]
    names += sorted((n for n in all_sheets if re.fullmatch(re.escape(period) + r'_\d+', n)),
                    key=lambda n: int(n.rsplit('_', 1)[-1]))
    schema = None
    column_names = None
    total = 0
    years = set()
    writers = {}
    buffers = defaultdict(list)
    mandatory = ['territory_id', 'territory_name'] + (['year'] if period == 'Annual' else ['year', 'month'] if period == 'Monthly' else ['period'])
    def flush(partition):
        if not buffers[partition]:
            return
        tbl = pa.Table.from_pylist(buffers[partition], schema=schema)
        writers[partition].write_table(tbl)
        buffers[partition].clear()
    try:
        for sheet in names:
            header, source_rows = _sheet_data(path, sheet)
            if not set(mandatory).issubset(header):
                raise ValueError(f'{level}:{sheet} missing keys {set(mandatory) - set(header)}')
            if column_names is None:
                column_names = header
                nonmetrics = set(mandatory)
                fields = []
                for col in header:
                    if col in ('territory_id', 'territory_name', 'period'):
                        t = pa.string()
                    elif col == 'year':
                        t = pa.int16()
                    elif col == 'month':
                        t = pa.int8()
                    else:
                        t = pa.float64()
                    fields.append(pa.field(col, t))
                schema = pa.schema(fields)
            elif header != column_names:
                raise ValueError(f'Incompatible headers between {period} sheets of {level}')
            for raw in source_rows:
                record = dict(zip(header, raw))
                tid = _to_clean_str(record.get('territory_id'))
                if not tid:
                    continue
                item = {}
                for col in header:
                    val = record.get(col)
                    if col in ('territory_id', 'territory_name', 'period'):
                        item[col] = _to_clean_str(val)
                    elif col in ('year', 'month'):
                        item[col] = _to_int(val)
                    else:
                        item[col] = _to_float(val)
                y = item.get('year') if period != 'Multiyear' else None
                if period != 'Multiyear' and not (1985 <= y <= 2025):
                    raise ValueError(f'Invalid year {y} in {sheet}:{level}')
                if period == 'Monthly' and not (1 <= item['month'] <= 12):
                    raise ValueError(f'Invalid month {item["month"]} in {sheet}:{level}')
                partition = str(y) if y is not None else 'all'
                years.add(partition)
                if partition not in writers:
                    file_path = output_dir / 'metrics' / level / period.lower() / f'year={partition}' / 'data.parquet'
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    writers[partition] = pq.ParquetWriter(str(file_path), schema=schema, compression='zstd')
                buffers[partition].append(item)
                if len(buffers[partition]) >= batch_size:
                    flush(partition)
                total += 1
    finally:
        for part in list(writers):
            flush(part)
            writers[part].close()
    return {'rows': total, 'years': sorted(years),
            'columns': [x for x in (column_names or []) if x not in mandatory],
            'sheets_read': names}


def build_metrics_from_xlsx(workdir='.', replace=True):
    """Convert REAL XLSX files to fast Parquet, partitioned by year and level."""
    import shutil
    workdir = Path(workdir)
    source = workdir / 'source_xlsx'
    output = workdir / 'data'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'title': 'MapBiomas Atmosphere | Brazil', 'source': 'MapBiomas Atmosphere GCS v01 statistics / XLSX presentation v02',
                'geometry_source': 'MapBiomas territory assets in GEE',
                'year_min': 1985, 'year_max': 2025, 'levels': {}, 'catalog_version': '1'}
    for level in TERRITORIES:
        path = source / workbook_name(level)
        if not path.exists():
            print(f'SKIP absent workbook: {level}')
            continue
        if replace:
            shutil.rmtree(output / 'metrics' / level, ignore_errors=True)
        dest = output / 'territories' / f'{level}.parquet'
        count = _write_territories(path, dest)
        periods = {}
        for p in ('Annual', 'Monthly', 'Multiyear'):
            periods[p] = _build_period(path, level, p, output)
        dictionaries = _catalog_from_xlsx(path)
        metadata_path = output / 'dictionaries' / f'{level}.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(dictionaries, ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['levels'][level] = {'label': LABELS[level], 'vector_asset': TERRITORIES[level],
                                     'territory_count': count, 'periods': periods}
        print(f'BUILT {level}: {count:,} territories; ' + ', '.join(f'{k} {v["rows"]:,}' for k,v in periods.items()))
    outpath = output / 'catalog.json'
    outpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'CATALOG: {outpath}; levels={len(manifest["levels"])}')
    return manifest


def _geometry_description(level):
    return f'MB_ATMOSPHERE_DASH_GEO_{level.upper()}'


def submit_geometry_exports(ee, workdir='.', overwrite=False, tolerance_m=None):
    """Start EE table export jobs for real simplified vector geometries to existing GCS bucket."""
    from google.cloud import storage
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    d = {'submitted': {}, 'skipped': [], 'already_running': []}
    recent_tasks = ee.batch.Task.list()
    for level, asset in TERRITORIES.items():
        if not (Path(workdir) / 'source_xlsx' / workbook_name(level)).exists():
            continue
        prefix = GEO_PREFIX + level
        existing_blobs = list(client.list_blobs(BUCKET, prefix=prefix))
        if existing_blobs and not overwrite:
            d['skipped'].append(level)
            print(f'GEOMETRY EXISTS: {level} (no EE task submitted)')
            continue
        desc = _geometry_description(level)
        active = [t for t in recent_tasks if t.config.get('description') == desc and t.status().get('state') in ('READY', 'RUNNING')]
        if active:
            d['already_running'].append(level)
            print(f'GEOMETRY ALREADY QUEUED: {level}')
            continue
        tolerance = (tolerance_m or {}).get(level, 500 if level == 'indigenous_lands' else 1000)
        fc = ee.FeatureCollection(asset)
        def clean(f):
            f = ee.Feature(f)
            simplified = f.geometry().simplify(maxError=tolerance)
            # EXACT same ID as original stats extraction: system:index
            return ee.Feature(simplified, {'territory_id': ee.String(f.get('system:index'))})
        cleaned = fc.map(clean)
        task = ee.batch.Export.table.toCloudStorage(
            collection=cleaned, description=desc, bucket=BUCKET,
            fileNamePrefix=prefix, fileFormat='GeoJSON'
        )
        task.start()
        d['submitted'][level] = task.id
        print(f'EE GEOMETRY SUBMITTED: {level} ({tolerance} m visual simplification), task={task.id}')
    return d


def wait_geometry_exports(ee, workdir='.', check_every=30, timeout_minutes=240):
    """Wait for geometry exports, fail explicitly if any export failed."""
    from google.cloud import storage
    client = storage.Client(project=PROJECT)
    levels = [k for k in TERRITORIES if (Path(workdir) / 'source_xlsx' / workbook_name(k)).exists()]
    stop = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < stop:
        states = {}
        recent_tasks = ee.batch.Task.list()
        for level in levels:
            if list(client.list_blobs(BUCKET, prefix=GEO_PREFIX + level)):
                states[level] = 'GCS_READY'
                continue
            matches = [t for t in recent_tasks if t.config.get('description') == _geometry_description(level)]
            if not matches:
                states[level] = 'MISSING_TASK'
            else:
                s = matches[0].status()
                states[level] = s.get('state', '?')
                if states[level] in ('FAILED', 'CANCELLED'):
                    raise RuntimeError(f'GEE geometry failed: {level}: {s}')
        print('GEOMETRY EXPORTS:', states)
        if all(s == 'GCS_READY' for s in states.values()):
            return states
        time.sleep(check_every)
    raise TimeoutError('Geometry exports not finished. Re-run wait_geometry_exports after checking EE Tasks.')


def download_geojson_exports(workdir='.', require_every_level=True):
    """Download real EE GeoJSON exports, normalize keys, QA against actual XLSX territory IDs."""
    from google.cloud import storage
    import pyarrow.parquet as pq
    client = storage.Client(project=PROJECT)
    workdir = Path(workdir)
    output = workdir / 'data' / 'geometries'
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for level in TERRITORIES:
        territory_parquet = workdir / 'data' / 'territories' / f'{level}.parquet'
        if not territory_parquet.exists():
            continue
        blobs = [b for b in client.list_blobs(BUCKET, prefix=GEO_PREFIX + level) if b.name.lower().endswith(('.geojson', '.json'))]
        if not blobs:
            if require_every_level:
                raise FileNotFoundError(f'Missing GeoJSON export for {level}: gs://{BUCKET}/{GEO_PREFIX + level}')
            print('SKIP missing geometry:', level)
            continue
        items = []
        for blob in blobs:
            source_geo = json.loads(blob.download_as_text(encoding='utf-8'))
            if source_geo.get('type') == 'FeatureCollection':
                features = source_geo.get('features', [])
            elif source_geo.get('type') == 'Feature':
                features = [source_geo]
            else:
                raise ValueError(f'Unexpected GeoJSON content: {blob.name}')
            for f in features:
                raw_id = f.get('properties', {}).get('territory_id')
                if raw_id is None:
                    continue
                tid = str(raw_id)
                if f.get('geometry') is not None:
                    items.append({'type': 'Feature', 'properties': {'territory_id': tid},
                                  'geometry': f['geometry']})
        if len(set(f['properties']['territory_id'] for f in items)) != len(items):
            raise ValueError(f'Duplicate geometry territory_id for {level}')
        table = pq.read_table(str(territory_parquet), columns=['territory_id'])
        ids = set(table.column('territory_id').to_pylist())
        geo_ids = set(f['properties']['territory_id'] for f in items)
        matched = len(ids & geo_ids)
        ratio = matched / max(1, len(ids))
        if ratio < .99:
            missing = sorted(ids-geo_ids)[:15]
            raise ValueError(f'Geometry ID mismatch in {level}: only {matched}/{len(ids)} matched. Sample missing={missing}. Check original GEE vector versions.')
        dest = output / f'{level}.geojson'
        dest.write_text(json.dumps({'type':'FeatureCollection','features':items},
                                    ensure_ascii=False,separators=(',',':')), encoding='utf-8')
        stats[level] = {'xlsx_territories': len(ids), 'features_in_geojson': len(items),
                        'matched': matched, 'match_rate': round(ratio,5), 'megabytes': round(dest.stat().st_size/1e6,2)}
        print(f'GEOMETRY QA {level}: {matched}/{len(ids)} matches, {dest.stat().st_size/1e6:.2f} MB')
    (workdir / 'data' / 'geometry_qa.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def validate_dataset(workdir='.'):
    import pyarrow.parquet as pq
    p=Path(workdir)/'data'
    cat=json.loads((p/'catalog.json').read_text(encoding='utf-8'))
    for level, v in cat['levels'].items():
        ids=set(pq.read_table(p/'territories'/f'{level}.parquet',columns=['territory_id']).column('territory_id').to_pylist())
        for period in ('Annual','Monthly','Multiyear'):
            info=v['periods'][period]
            if info['rows'] <= 0:
                raise ValueError(f'Empty period {level} {period}')
            files=list((p/'metrics'/level/period.lower()).glob('year=*/data.parquet'))
            if len(files)!=len(info['years']):
                raise ValueError(f'Incomplete Parquet period {level} {period}: {len(files)} vs {len(info["years"])}')
            actual_rows=sum(pq.ParquetFile(str(file)).metadata.num_rows for file in files)
            if actual_rows != info['rows']:
                raise ValueError(f'Row count mismatch for {level} {period}: {actual_rows} vs {info["rows"]}')
            for sample in (files[0],files[-1]):
                schema=pq.read_schema(sample)
                if not all(c in schema.names for c in ['territory_id','territory_name']):
                    raise ValueError(f'Bad table columns {sample}')
        geom=p/'geometries'/f'{level}.geojson'
        if not geom.exists():
            print(f'GEOMETRY NOT AVAILABLE: {level}, map will be disabled for this level')
    print('DATASET VALIDATED:',len(cat['levels']),'levels')
    return cat


if __name__ == '__main__':
    print('Import from Colab; see README.md for exact steps.')

