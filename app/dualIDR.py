import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file
from bokeh.events import MouseWheel, SelectionGeometry, ButtonClick, MouseMove, Reset, Press
from bokeh.io import curdoc
from bokeh.models import Div, ColumnDataSource, LabelSet, HoverTool, CategoricalColorMapper, Slider, Button, HoverTool, TextInput, TextAreaInput, CheckboxButtonGroup, CheckboxGroup
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import row, column, layout
import time
import warnings
import sys
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
warnings.filterwarnings("ignore")

sys.path.append('utils')
from getData import get_data

selCol  	= []
selCol_idx  = []
selFil  	= []
selFil_idx  = []

# global
early_exaggeration = 1

# sliders
slider_perplexity_1 = Slider(start=2, end=200, value=130, step= 1, title="perplexity")
slider_perplexity_2 = Slider(start=2, end=200, value=130, step= 1, title="perplexity")
slider_lr_1 = Slider(start=-5, end=5, value=2,    step=.1, title="learning_rate (log)")
slider_lr_2 = Slider(start=-5, end=5, value=2,    step=.1, title="learning_rate (log)")

def get_idx_selection(text,lista_labels, no_empty=True):
	'''
	select all items from "lista_labels" 
	that contain any of the substrings of "text" 
	separated by commas

	text:
		user introduced string with a list of items

	lista_labels:
		list with the labels from which the indices will be selected

	no_empty: 
		if it is True, everything will be selected in case text is empty

	'''

	idx_sel = []
	if len(text)>0:
		items = text.replace(' ','').split(',')
		# 2023-09-09: se eligen terminos de la lista de cadenas "literamente iguales" (no "incluidas en")
		# aux = list(map(lambda y:[lista_labels.index(X) for X in lista_labels if y in X],items))
		aux = list(map(lambda y:[lista_labels.index(X) for X in lista_labels if y == X],items))
		idx_sel = list(set().union(*aux))
	elif no_empty:
		idx_sel = [i for i,v in enumerate(lista_labels)]
	
	return idx_sel


def update_affinities():
	global affinities_1, affinities_2
	global embedding_1, embedding_2
	global y1, y2

	print('computing affinities (samples, genes) ...')
	affinities_1 = affinity.PerplexityBasedNN(
	    x1[:,idx_var],
	    perplexity=slider_perplexity_1.value,
	    metric="euclidean",
	    n_jobs=8,
	    random_state=42,
	    verbose=False,
	)

	affinities_2 = affinity.PerplexityBasedNN(
	    x2[:,idx_obs],
	    perplexity=slider_perplexity_2.value,
	    metric="euclidean",
	    n_jobs=8,
	    random_state=42,
	    verbose=False,
	)

	print('creating embedding objects (samples, genes) ...')
	# DEFINIMOS OBJETO EMBEDDING PARA OPTIMIZAR
	embedding_1 = TSNEEmbedding(
	    y1,
	    affinities_1,
	    negative_gradient_method="bh",
	    n_jobs=8,
	    verbose=False,
	)

	# DEFINIMOS OBJETO EMBEDDING PARA OPTIMIZAR
	embedding_2 = TSNEEmbedding(
	    y2,
	    affinities_2,
	    negative_gradient_method="bh",
	    n_jobs=8,
	    verbose=False,
	)
	print('... done')

# Obtención de datos

F, Fy, Fx = get_data()

X = F
M = F.shape[0]
N = F.shape[1]

selFil_idx = np.arange(M)
selCol_idx = np.arange(N)

idx_obs = np.arange(M)
idx_var = np.arange(N)

# muestras
x1 = X.copy()
y1 = np.random.randn(x1.shape[0],2)

# armónicos
x2 = X.T.copy()
y2 = np.random.randn(x2.shape[0],2)

# 2D map initialization
print('initializing maps (samples, genes) ...')
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=1)
y1 = pca.fit_transform(X)
y2 = pca.fit_transform(X.T)

N = x1.shape[0]		# número de muestras
M = x2.shape[0]		# número de genes

# selection colors
colors_obs = ['#000000']*N
colors_var = ['#000000']*M

# column datas sources
source_obs = ColumnDataSource({'x1':y1[:,0],'y1':y1[:,1],'labels':Fy,'colors':colors_obs})
source_var = ColumnDataSource({'x2':y2[:,0],'y2':y2[:,1],'labels':Fx,'colors':colors_var})

# compute affinities by first time
update_affinities()

def update_tsne():
	"""update conditional t-SNE's of samples and genes
	samples: distances conditioned to a subset of the genes (idx_var)
	  genes: distances conditioned to a subset of the samples (idx_obs)

	Notes:
	  - the update takes just a few epochs
	  - by default, early_exaggeration=1 (non distorted basic tSNE minimization)
	"""
	global x1,x2
	global y1,y2
	global embedding_1, embedding_2

	embedding_1 = embedding_1.optimize(n_iter=1, exaggeration=early_exaggeration, momentum=0.5,learning_rate=np.exp(slider_lr_1.value),verbose=False)
	embedding_2 = embedding_2.optimize(n_iter=1, exaggeration=early_exaggeration, momentum=0.5,learning_rate=np.exp(slider_lr_2.value),verbose=False)

	y1 = np.array(embedding_1)
	y2 = np.array(embedding_2)

	source_obs.data['x1'] = y1[:,0]
	source_obs.data['y1'] = y1[:,1]
	source_var.data['x2'] = y2[:,0]
	
def reset_original_state():
	global y1,y2
	# train a UMAP
	print('training UMAP (samples and genes)...')
	y1 = pca.fit_transform(X)
	y2 = pca.fit_transform(X.T)

	# set parameters to their default values
	print('resetting DR parameters to defaults...')
	slider_perplexity_1.value = 20
	slider_lr_1.value = 1

	# select all samples and genes (default)
	textinput_colorsel_obs.value = ''
	textinput_colorsel_var.value = ''

	# update the selection
	update_conditional_dr()
	
def selection_callback_2Dfil(event):
	global selFil, selFil_idx
	selFil = [Fy[indice] for indice in source_obs.selected.indices]
	selFil_idx = source_obs.selected.indices
	textinput_colorsel_obs.value = ','.join([Fy[i] for i in selFil_idx])

def selection_callback_2Dcol(event):
	global selCol, selCol_idx
	selCol = [Fx[indice] for indice in source_var.selected.indices]
	selCol_idx = source_var.selected.indices
	textinput_colorsel_var.value = ','.join([Fx[i] for i in selCol_idx])

def update_conditional_dr():
	global idx_obs, idx_var
	global x1, x2

	idx_obs = get_idx_selection(textinput_colorsel_obs.value,Fy,no_empty=True)
	idx_var = get_idx_selection(textinput_colorsel_var.value,Fx,no_empty=True)

	# show in textboxes the interpreted readings (for user check)
	textinput_colorsel_obs.value = ','.join([Fy[i] for i in idx_obs])
	textinput_colorsel_var.value = ','.join([Fx[i] for i in idx_var])

	update_affinities()

# COLOR OF SAMPLES AND GENES
def update_color_obs_1():
	idx = get_idx_selection(textinput_colorsel_obs.value,Fy,no_empty=True)
	source_obs.data['colors'] = [textinput_color_obs.value if i in idx else source_obs.data['colors'][i] for i in range(N)]

def update_color_var_1():
	idx = get_idx_selection(textinput_colorsel_var.value,Fx,no_empty=True)
	source_var.data['colors'] = [textinput_color_var.value if i in idx else source_var.data['colors'][i] for i in range(M)]

def update_color_obs(attr,old,new):
	idx = get_idx_selection(new,Fx)
	
	# gene expression color scale
	if len(idx)==1:
		from matplotlib.cm import seismic
		from matplotlib.colors import rgb2hex
		textinput_color_obs.value = Fx[idx[0]]
		source_obs.data['colors'] = [rgb2hex(seismic(np.interp(x,[-1,1],[0,1]))) for x in x1[:,idx[0]]]
	else:
		update_color_obs_1()
	print(attr,old,new)

def update_color_var(attr,old,new):
	idx = get_idx_selection(new,Fy)
	
	# gene expression color scale
	if len(idx)==1:
		from matplotlib.cm import seismic
		from matplotlib.colors import rgb2hex
		textinput_color_var.value = Fy[idx[0]]
		source_var.data['colors'] = [rgb2hex(seismic(np.interp(x,[-1,1],[0,1]))) for x in x2[:,idx[0]]]
	else:
		update_color_var_1()

def checkbox_callback(attr):
	if 0 in attr:
		labels_obs.visible = True
	else:
		labels_obs.visible = False

	if 1 in attr:
		labels_var.visible = True
	else:
		labels_var.visible = False

def slider_perplexity_1_callback(att,old,new):
	update_affinities()
slider_perplexity_1.on_change('value',slider_perplexity_1_callback)

def slider_perplexity_2_callback(att,old,new):
	update_affinities()
slider_perplexity_2.on_change('value',slider_perplexity_2_callback)


# buttons
button_conditional_dr = Button(label='use selected genes & samples')
button_conditional_dr.on_click(update_conditional_dr)
button_obs = Button(label='selection (smp) ➡ color')
button_obs.on_click(update_color_obs_1)
button_var = Button(label='selection (gen) ➡ color')
button_var.on_click(update_color_var_1)
button_reset = Button(label='reset projections')
button_reset.on_click(reset_original_state)


def change_exaggeration():
	global early_exaggeration

	if button_exaggeration.label=='exaggeration = ON':
		button_exaggeration.label = 'exaggeration = OFF'
		early_exaggeration = 1
	else:
		button_exaggeration.label = 'exaggeration = ON'
		early_exaggeration = 2

button_exaggeration = Button(label='exaggeration = OFF')
button_exaggeration.on_click(change_exaggeration)



# textinput boxes
textinput_colorsel_obs = TextAreaInput(value="", title='selected samples', width=500,height=145,max_length=10000)
textinput_colorsel_obs.value = ''

textinput_colorsel_var = TextAreaInput(value="", title='selected genes', width=500,height=145,max_length=10000)
textinput_colorsel_var.value = ''

textinput_color_obs = TextInput(title='color for samples',width=145)
textinput_color_obs.value = 'black'
textinput_color_obs.on_change('value',update_color_obs)

textinput_color_var = TextInput(title='color for genes',width=145)
textinput_color_var.value = 'black'
textinput_color_var.on_change('value',update_color_var)


checkbox_group = CheckboxGroup(labels=['sample labels', 'gene labels'], active=[0,1])
checkbox_group.on_click(checkbox_callback)


# ELEMENTO DE TEXTO
cabecera = Div(text='''
	<h1>Dual iDR de líneas celulares en normoxia/hipoxia (PGL184, PCC64 y PCC66)</h1>
			   <p><img src="https://gsdpi.edv.uniovi.es/logo-gsdpi-research-team.png", width="100px"> <i>Grupo de Supervisión, Diagnóstico y Descubrimiento del Conocimiento en Procesos de Ingeniería (GSDPI)</i>. Universidad de Oviedo, 2023</p>
	<p><i>Descripción</i>: Aplicación de reducción de la dimensionalidad <i>dual</i> e <i>interactiva</i>. Muestra dos proyecciones (<i>dual</i>) actualizadas en tiempo real, la "sample view" y la "gene view", 
			   en las que se organizan espacialmente las muestras y los genes según la similitud en su expresión genética para grupos concretos de genes y muestras respectivaemente. El usuario 
			   puede condicionar las proyecciones en tiempo de ejecución cambiando los grupos de genes y/o de muestras cuyas expresiones se tienen en cuenta en las proyecciones. Los cambios son
			   reflejados en tiempo real y de forma continua. La aplicación permite también asignar colores a grupos de genes o de muestras, permitiendo al usuario hacer un seguimiento (<i>"tracking"</i>) de elementos de interés.</p>''',width=1000,height=250)


# SAMPLE VIEW
fig1 = figure(width=700,height=700,title='sample view 🧪',
	tools="crosshair,lasso_select,pan,reset,wheel_zoom",
	tooltips=[("sample:","@labels"),("cancer:","@can_labels")],match_aspect=True,output_backend='webgl')
fig1.circle(x='x1',y='y1',color='colors', source=source_obs,size=7)
fig1.title.text_font_size='16px'
fig1.on_event(SelectionGeometry,selection_callback_2Dfil)
labels_obs = LabelSet(x='x1',y='y1',text='labels',source = source_obs,y_offset=5)
labels_obs.text_alpha=1
labels_obs.text_align='center'
labels_obs.text_font_size={'value': '10px'}
fig1.add_layout(labels_obs)

# GENE VIEW
fig2 = figure(width=700,height=700,title='gene view 🧬',
	tools="crosshair,lasso_select,pan,reset,wheel_zoom",
	tooltips=[("gene:","@labels")],
	match_aspect=True,output_backend='webgl')
fig2.circle(x='x2',y='y2',color='colors',source=source_var,size=7)
fig2.title.text_font_size='16px'
fig2.on_event(SelectionGeometry,selection_callback_2Dcol)
labels_var = LabelSet(x='x2',y='y2',text='labels',source = source_var)
labels_var.text_alpha=1
labels_var.text_align='center'
labels_var.text_font_size={'value': '10px'}
fig2.add_layout(labels_var)


curdoc().add_periodic_callback(update_tsne,200)		# callback lenta

curdoc().add_root(layout(column(cabecera, row(
	column(fig1,slider_perplexity_1,slider_lr_1,textinput_color_obs,textinput_colorsel_obs),
	column(fig2,slider_perplexity_2,slider_lr_2,textinput_color_var,textinput_colorsel_var),
	column(button_conditional_dr,button_reset,button_obs,button_var,checkbox_group,button_exaggeration)
	))))

