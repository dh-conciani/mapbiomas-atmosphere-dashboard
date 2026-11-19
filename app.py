"""MapBiomas Atmosphere Explorer — Streamlit app using genuine XLSX-derived data
and authentic GEE geometry exports. The live website needs no EE/GCS credentials.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent / 'data'
PERIODS = ['Annual', 'Monthly', 'Multiyear']
COLORS = {'Atmosphere C2':'#3478BD','Land Cover and Land Use':'#2E8B57',
          'Themes (Fire, Water, Logging)':'#E18A3C'}

st.set_page_config(page_title='MapBiomas Atmosphere Explorer',page_icon='🌎',layout='wide',
                   initial_sidebar_state='expanded')
st.markdown('''
<style>
.block-container {padding-top: 1.3rem; max-width: 1600px;}
h1, h2, h3 {letter-spacing:-.025em;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
[data-testid="stSidebar"] {background: #111c2c10;}
</style>
''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def catalog():
    path=BASE/'catalog.json'
    return json.loads(path.read_text(encoding='utf-8'))

@st.cache_data(show_spinner=False)
def get_territories(level):
    p=BASE/'territories'/f'{level}.parquet'
    return pd.read_parquet(p).fillna('')

@st.cache_data(show_spinner=False)
def get_dictionary(level):
    p=BASE/'dictionaries'/f'{level}.json'
    return json.loads(p.read_text(encoding='utf-8'))

@st.cache_data(show_spinner=False)
def get_geojson(level):
    p=BASE/'geometries'/f'{level}.geojson'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

@st.cache_data(show_spinner='Reading selected data...',max_entries=32)
def load_metrics(level,period,years,selected_columns,selected_ids):
    root=BASE/'metrics'/level/period.lower()
    files=[root/f'year={y}'/'data.parquet' for y in years]
    files=[f for f in files if f.exists()]
    keys=['territory_id','territory_name']
    keys += (['year','month'] if period=='Monthly' else ['year'] if period=='Annual' else ['period'])
    desired=list(dict.fromkeys(keys+list(selected_columns)))
    if not files:
        return pd.DataFrame(columns=desired)
    frames=[pd.read_parquet(f,columns=desired,filters=[("territory_id","in",list(selected_ids))]) for f in files]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=desired)


def metric_label(metric, metadata):
    unit=metadata.get(metric,{}).get('unit','')
    return f'{metric}  [{unit}]' if unit else metric


def _polygon_parts(geometry):
    """Yield polygon coordinate arrays from valid GeoJSON polygonal geometry.

    Earth Engine GeoJSON may contain nested GeometryCollections, which expose
    `geometries` instead of `coordinates`. Plotly maps need Polygon/MultiPolygon.
    Non-polygonal members (points/lines) are intentionally not mapped as fills.
    """
    if not isinstance(geometry, dict):
        return
    geom_type = geometry.get('type')
    if geom_type == 'Polygon':
        coordinates = geometry.get('coordinates')
        if isinstance(coordinates, list) and coordinates:
            yield coordinates
    elif geom_type == 'MultiPolygon':
        for polygon in geometry.get('coordinates') or []:
            if isinstance(polygon, list) and polygon:
                yield polygon
    elif geom_type == 'GeometryCollection':
        for child in geometry.get('geometries') or []:
            yield from _polygon_parts(child)


def polygonal_geometry(geometry):
    """Normalize polygons and polygon-containing GeometryCollections for Plotly.

    Coordinates are reused unmodified; no geometric simplification or
    aggregation of metric values occurs here.
    """
    polygons = list(_polygon_parts(geometry))
    if not polygons:
        return None
    if len(polygons) == 1:
        return {'type': 'Polygon', 'coordinates': polygons[0]}
    return {'type': 'MultiPolygon', 'coordinates': polygons}


def geo_bounds(features):
    """Return (center, zoom) from polygon geometries; tolerate absent geometry."""
    xs, ys = [], []

    def walk(coordinates):
        if not isinstance(coordinates, (list, tuple)) or not coordinates:
            return
        if (len(coordinates) >= 2
                and isinstance(coordinates[0], (int, float))
                and isinstance(coordinates[1], (int, float))
                and np.isfinite(coordinates[0])
                and np.isfinite(coordinates[1])):
            xs.append(float(coordinates[0]))
            ys.append(float(coordinates[1]))
            return
        for child in coordinates:
            walk(child)

    for feature in features:
        geometry = polygonal_geometry(feature.get('geometry'))
        if geometry is not None:
            walk(geometry['coordinates'])

    if not xs:
        return {'lat': -12, 'lon': -54}, 4
    center = {'lat': (min(ys) + max(ys)) / 2,
              'lon': (min(xs) + max(xs)) / 2}
    extent = max(max(xs) - min(xs), (max(ys) - min(ys)) * 1.6, 1)
    zoom = float(np.clip(np.log2(360 / extent) - 1.25, 2.0, 10.5))
    return center, zoom

def numberfmt(m,meta):
    grp=meta.get(m,{}).get('group','')
    return ',.0f' if ('Land Cover' in grp or 'Themes' in grp) else ',.2f'


st.title('🌎 MapBiomas Atmosphere Explorer')
st.caption('Brazil · Collection 2 atmospheric indicators + land cover + fire/water themes · 1985–2025')
if not (BASE/'catalog.json').exists():
    st.error('Dataset not bundled yet. Run the **Build Dashboard Dataset** Colab, export GEE geometries, and copy its generated data/ directory into the GitHub repository.')
    st.stop()
manifest=catalog()
levels=manifest.get('levels',{})
if not levels:
    st.error('Catalog contains no territory levels.'); st.stop()
with st.sidebar:
    st.header('Explore the data')
    level=st.selectbox('Territorial level',list(levels),format_func=lambda k:levels[k]['label'])
    available=[p for p in PERIODS if levels[level]['periods'][p]['rows']>0]
    period=st.radio('Temporal resolution',available,horizontal=True)
    info=levels[level]['periods'][period]
    areas=get_territories(level)
    areas['territory_id']=areas['territory_id'].astype(str)
    areas['territory_name']=areas['territory_name'].astype(str)
    areas=areas.drop_duplicates(subset=['territory_id'])
    counts=areas['territory_name'].value_counts().to_dict()
    label_of={r.territory_id:(r.territory_name + (f' · {r.territory_id}' if counts[r.territory_name]>1 else '')) for r in areas[['territory_id','territory_name']].itertuples(index=False)}
    options=areas['territory_id'].tolist()
    initial=options if len(options)<=30 else options[:10]
    selection_key=f'selected_territories_{level}'
    if selection_key not in st.session_state:
        st.session_state[selection_key]=initial
    def set_selection(value):
        st.session_state[selection_key]=list(value)
    a,b=st.columns(2)
    with a:
        st.button('Select all',use_container_width=True,on_click=set_selection,args=(options,),key=f'selectall_{level}')
    with b:
        st.button('Clear',use_container_width=True,on_click=set_selection,args=([],),key=f'clear_{level}')
    pick=st.multiselect('Units within this level',options,key=selection_key,
                        format_func=lambda i:label_of.get(i,i),
                        help='Select any state, biome, region or indigenous land. IDs link precisely to GEE polygons.')
    years_available=sorted(int(s) for s in info['years'] if s!='all')
    if period!='Multiyear':
        year_range=st.slider('Year interval',min(years_available),max(years_available),
                              (min(years_available),max(years_available)))
        years=[y for y in years_available if year_range[0]<=y<=year_range[1]]
        if period=='Monthly':
            months=st.multiselect('Months',list(range(1,13)),default=list(range(1,13)),
                                  format_func=lambda m:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
        else:months=[]
    else:
        years=['all'];months=[]
    dictionary=get_dictionary(level)
    meta={r.get('column_name'):r for r in dictionary if r.get('period')==period}
    metrics=info.get('columns',[])
    default=[m for m in ['mean_temperature_c','precipitation_mm','forest_area_ha','burned_area_ha'] if m in metrics]
    if not default:default=metrics[:min(2,len(metrics))]
    variable_key=f'variables_{level}_{period}'
    if variable_key not in st.session_state:
        st.session_state[variable_key]=default
    def set_variables(values):
        st.session_state[variable_key]=list(values)
    var_a,var_b,var_c=st.columns(3)
    with var_a:
        st.button('All climate',use_container_width=True,key=f'climate_{level}_{period}',
                  on_click=set_variables,args=([m for m in metrics if meta.get(m,{}).get('group')=='Atmosphere C2'],))
    with var_b:
        st.button('All land/themes',use_container_width=True,key=f'land_{level}_{period}',
                  on_click=set_variables,args=([m for m in metrics if meta.get(m,{}).get('group')!='Atmosphere C2'],))
    with var_c:
        st.button('All variables',use_container_width=True,key=f'everything_{level}_{period}',
                  on_click=set_variables,args=(metrics,))
    selected=st.multiselect('Variables — select as many as you want',metrics,key=variable_key,
                            format_func=lambda m:metric_label(m,meta),
                            help='Multiple variables appear together in charts, scatterplots and correlation matrices.')
    st.caption(f'{len(options):,} territorial units · {len(metrics)} metrics in {period.lower()} data')
    if st.checkbox('Show source descriptions',value=False):
        st.dataframe(pd.DataFrame([meta.get(m,{}) for m in selected]),hide_index=True)

if not pick:
    st.info('Choose at least one territorial unit in the sidebar to begin.');st.stop()
if not selected:
    st.info('Choose at least one variable in the sidebar.');st.stop()
if period=='Monthly' and not months:
    st.info('Choose at least one month.');st.stop()

frames=load_metrics(level,period,tuple(years),tuple(selected),tuple(pick))
frames['territory_id']=frames['territory_id'].astype(str)
filtered=frames[frames.territory_id.isin(set(pick))].copy()
if period=='Monthly':filtered=filtered[filtered.month.isin(months)].copy()
if filtered.empty:
    st.warning('No observations for this selection.');st.stop()
filtered['territory_label']=filtered['territory_id'].map(label_of)

k1,k2,k3,k4=st.columns(4)
k1.metric('Territorial units',f'{filtered.territory_id.nunique():,}')
k2.metric('Observations',f'{len(filtered):,}')
k3.metric('Variables selected',str(len(selected)))
k4.metric('Period','Multiyear' if period=='Multiyear' else f'{year_range[0]}–{year_range[1]}')

st.divider()
st.subheader('Interactive map')
map_col,details_col=st.columns([3.25,1.15],gap='large')
with details_col:
    map_var=st.selectbox('Mapped variable',selected,format_func=lambda x:metric_label(x,meta))
    if period!='Multiyear':
        map_year=st.select_slider('Map year',options=years,value=years[-1])
        if period=='Monthly':
            map_month=st.selectbox('Map month',months,index=len(months)-1,
                                   format_func=lambda m:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1])
        else:map_month=None
    else:
        map_year=None;map_month=None
    st.caption('The colors correspond to one variable and one period. Select other variables in the charts below to compare them.')

geo=get_geojson(level)
with map_col:
    if geo is None:
        st.warning('Real GEE geometries have not been exported for this level. Run the geometry-export cell in the Colab. Charts and tables remain available.')
    else:
        map_data=filtered
        if map_year is not None:map_data=map_data[map_data.year==map_year]
        if map_month is not None:map_data=map_data[map_data.month==map_month]
        map_data=map_data[['territory_id','territory_label',map_var]].drop_duplicates('territory_id')
        value_ids=set(map_data.territory_id)
        matching_feats=[f for f in geo.get('features', []) if str(f.get('properties',{}).get('territory_id','')) in value_ids]
        selected_feats=[]
        skipped_geometries=0
        for feature in matching_feats:
            polygon_geometry=polygonal_geometry(feature.get('geometry'))
            if polygon_geometry is None:
                skipped_geometries+=1
                continue
            selected_feats.append({'type':'Feature',
                                   'properties':feature.get('properties',{}),
                                   'geometry':polygon_geometry})
        if skipped_geometries:
            st.warning(f'{skipped_geometries:,} geometries without usable polygon coordinates were skipped in the map. Statistics and charts remain available.')
        if len(selected_feats)>450:
            st.info(f'{len(selected_feats):,} polygons selected. Showing the first 450 on the map for browser performance; filter the units to view the rest. Charts retain all selected data.')
            selected_feats=selected_feats[:450]
            map_data=map_data[map_data.territory_id.isin({f['properties']['territory_id'] for f in selected_feats})]
        if not selected_feats:
            st.error('No GEE geometry IDs match the selected XLSX rows. Regenerate/validate the geometries in Colab.')
        else:
            center,zoom=geo_bounds(selected_feats)
            vals=pd.to_numeric(map_data[map_var],errors='coerce')
            midpoint=0 if vals.min(skipna=True)<0<vals.max(skipna=True) else None
            fig=px.choropleth_map(map_data,
                geojson={'type':'FeatureCollection','features':selected_feats},
                locations='territory_id',featureidkey='properties.territory_id',
                color=map_var,hover_name='territory_label',
                hover_data={'territory_id':True,map_var:':.2f'},
                color_continuous_scale='RdBu_r' if midpoint is not None else 'Viridis',
                color_continuous_midpoint=midpoint,
                map_style='carto-positron',center=center,zoom=zoom,opacity=.75,
                labels={map_var:metric_label(map_var,meta)})
            fig.update_traces(marker_line_width=.6,marker_line_color='rgba(22,45,65,.7)')
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0),height=540,coloraxis_colorbar=dict(title=meta.get(map_var,{}).get('unit','')))
            st.plotly_chart(fig,use_container_width=True)

st.divider()
st.subheader('Time-series and multi-variable comparisons')
chart_type=st.radio('Comparison style',
    ['Raw values — separate scales by unit','Standardized (z-scores) — compare shapes'],
    horizontal=True,help='Never directly compare °C, hectares and millimetres on one numerical axis. Z-scores show relative changes.')
ids=['territory_id','territory_label'] + (['year','month'] if period=='Monthly' else ['year'] if period=='Annual' else ['period'])
long=filtered.melt(id_vars=ids,value_vars=selected,var_name='variable',value_name='value')
long['value']=pd.to_numeric(long['value'],errors='coerce')
long=long.dropna(subset=['value'])
long['unit']=long['variable'].map(lambda m:meta.get(m,{}).get('unit','unverified') or 'unverified')
long['series']=long['territory_label']+' · '+long['variable']
if period=='Monthly':
    long['time']=pd.to_datetime(dict(year=long.year.astype(int),month=long.month.astype(int),day=1))
elif period=='Annual':
    long['time']=long.year.astype(int)
if period!='Multiyear':
    if chart_type.startswith('Standardized'):
        means=long.groupby(['territory_id','variable'])['value'].transform('mean')
        stds=long.groupby(['territory_id','variable'])['value'].transform('std').replace(0,np.nan)
        long['plot_value']=(long['value']-means)/stds
        fig=px.line(long,x='time',y='plot_value',color='territory_label',line_dash='variable',
                    hover_data=['variable','value','unit'],labels={'plot_value':'Z-score','time':'Date / Year'},
                    title='Standardized trajectories — selected units and variables')
        fig.update_yaxes(title='Z-score (unitless)')
        fig.update_layout(height=520,hovermode='x unified')
    else:
        units=sorted(long.unit.unique())
        fig=px.line(long,x='time',y='value',color='territory_label',line_dash='variable',
                    facet_row='unit',facet_row_spacing=min(.08,.25/max(1,len(units)-1)),
                    labels={'value':'Value','time':'Date / Year'},
                    title='Raw measurements — one axis per measurement unit',
                    height=max(470,245*len(units)))
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace('unit=','')))
        fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig,use_container_width=True)
else:
    fig=px.bar(long,x='territory_label',y='value',color='variable',barmode='group',
               facet_row='unit',height=max(450,270*long.unit.nunique()),
               title='Multiyear trend values by territory and variable')
    fig.for_each_annotation(lambda a:a.update(text=a.text.replace('unit=','')))
    fig.update_layout(xaxis_title='Territory',yaxis_title='Value')
    st.plotly_chart(fig,use_container_width=True)

st.divider()
left,right=st.columns([1,1],gap='large')
with left:
    st.subheader('Cross-variable scatter')
    if len(selected)>=2:
        x=st.selectbox('X variable',selected,index=0,key='scatter_x')
        y=st.selectbox('Y variable',selected,index=1,key='scatter_y')
        scatter=filtered[['territory_label','territory_id',x,y]+(['year','month'] if period=='Monthly' else ['year'] if period=='Annual' else [])].copy()
        scatter=scatter.dropna(subset=[x,y])
        if x==y:
            st.caption('Select different variables to see their relationship.')
        else:
            fig=px.scatter(scatter,x=x,y=y,color='territory_label',
                           hover_data=['territory_id']+(['year','month'] if period=='Monthly' else ['year'] if period=='Annual' else []),
                           labels={x:metric_label(x,meta),y:metric_label(y,meta)},opacity=.6)
            fig.update_layout(height=420,margin=dict(t=20))
            st.plotly_chart(fig,use_container_width=True)
    else:st.info('Select at least two variables to create a scatterplot.')
with right:
    st.subheader('Correlation matrix')
    if len(selected)>=2:
        corr=filtered[selected].apply(pd.to_numeric,errors='coerce').corr(min_periods=4)
        if corr.notna().any().any():
            fig=px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,
                          labels=dict(color='Pearson r'),aspect='auto')
            fig.update_layout(height=max(420,25*len(selected)),margin=dict(t=20))
            st.plotly_chart(fig,use_container_width=True)
        else:st.info('Not enough independent observations to compute correlations.')
    else:st.info('Select at least two variables.')

with st.expander('View data and download CSV',expanded=False):
    displayed=filtered[['territory_id','territory_name']+(['year','month'] if period=='Monthly' else ['year'] if period=='Annual' else ['period'])+selected].copy()
    st.dataframe(displayed,use_container_width=True,hide_index=True,height=360)
    st.download_button('Download filtered data (.csv)',
        displayed.to_csv(index=False).encode('utf-8-sig'),
        file_name=f'mapbiomas_atmosphere_{level}_{period.lower()}_selected.csv',mime='text/csv')
with st.expander('Metric dictionary and source provenance'):
    d=pd.DataFrame([r for r in dictionary if r.get('period')==period and r.get('column_name') in selected])
    st.dataframe(d,use_container_width=True,hide_index=True)
    st.caption('The data are original XLSX v02 measurements converted to Parquet without scaling or aggregation; polygons are simplified exports of original GEE territory assets. Correlations and z-scores are computed only for visualization.')
st.caption('MapBiomas Atmosphere · Community dashboard · Original collection data retain their own licensing and attribution.')

