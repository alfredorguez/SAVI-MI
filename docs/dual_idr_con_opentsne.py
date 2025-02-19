# PRUEBA SOM NUMBA CON DATOS SINTÉTICOS

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
from openTSNE import TSNEEmbedding
from openTSNE import affinity
from openTSNE import initialization
warnings.filterwarnings("ignore")


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



def disambiguate(strings):
  print('disambiguating symbols')
  counts = {}
  disambiguated = []
  
  for s in strings:
    if s not in counts:
      counts[s] = 0
    counts[s] += 1
    if counts[s] > 1:
      disambiguated.append(f"{s}-{counts[s]}") 
    else:
      disambiguated.append(s)
        
  return disambiguated


def get_data():
	print('importando matriz de expresiones ...')
	X = pd.read_excel('RNAseq lineas PPGL normalizados todos.xlsx',skiprows=2,usecols='E:AE',header=None).values.T

	print('normalizando expresiones de X (división por max abs)')
	X = X/np.max(np.abs(X))

	print('importando anotaciones de observaciones ...')
	# aux = pd.read_excel('RNAseq lineas PPGL normalizados todos.xlsx',skiprows=1,usecols='E:AE',nrows=1,header=None).values
	aux = pd.read_excel('RNAseq lineas PPGL normalizados todos.xlsx',skiprows=0,usecols='E:AE',nrows=2,header=None).fillna(method='ffill',axis=1).values
	ann_obs = []
	ann_obs.append(aux[0,:].tolist())
	ann_obs.append(aux[1,:].tolist())

	print('importando anotaciones de variables ...')
	# aux = pd.read_excel('RNAseq lineas PPGL normalizados todos.xlsx',skiprows=2,usecols='C:D',header=None).values
	# ann_var = []
	# ann_var.append(aux[:,0].tolist())
	# ann_var.append(aux[:,1].tolist())
	aux = pd.read_excel('RNAseq lineas PPGL normalizados todos.xlsx',skiprows=2,usecols='A',header=None).values.tolist()
	ann_var = []
	ann_var.append(              [v[0].split(',')[2] for i,v in enumerate(aux)] )
	ann_var.append( disambiguate([v[0].split(',')[1] for i,v in enumerate(aux)]))
 
	data = data_class(X, ann_obs, ann_var)

	return data


class data_class:

    def __init__(self, X, obs, var):
        self.X = X
        self.obs = obs
        self.var = var

    def select(self, idx_rows=None, idx_cols=None):
        
        if idx_rows is None and idx_cols is None:
            return self
        
        X_sel = self.X
        if idx_rows is not None:
            X_sel = X_sel[idx_rows,:] 
        if idx_cols is not None:
            X_sel = X_sel[:,idx_cols]
            
        obs_sel = self.obs
        if idx_rows is not None:
            obs_sel = [[self.obs[i][row] for row in idx_rows] for i in range(len(self.obs))]
            
        var_sel = self.var    
        if idx_cols is not None:
            var_sel = [[self.var[j][col] for col in idx_cols] for j in range(len(self.var))]
        
        return data_class(X_sel, obs_sel, var_sel)

    def get_shape(self):
        return self.X.shape
    
    def get_obs_shape(self):
        return len(self.obs), len(self.obs[0])

    def get_var_shape(self):
        return len(self.var), len(self.var[0])

    def export_to_dict(self):
        M, N = self.get_shape()
        return {'X': self.X, 'obs': self.obs, 'var': self.var, 'M': M, 'N': N}




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

# OBTENEMOS DATOS DE LOS xlsx
data = get_data()


# SELECCIONAMOS GENES DE HIPOXIA (¡ podrían ser cualesquiera otros de los 18000+ !)
# hipoxia gene list
hypoxia_35      = ['ACTL8', 'AK3L1', 'EGLN3', 'GPR139', 'KISS1R', 'PFKL', 'PKM2', 'SLC16A3', 'SLC1A6', 'ADORA2A', 'BNIP3', 'C11orf88', 'C8orf58', 'CCDC19', 'CCDC64B', 'CYB5A', 'DGCR5', 'EGFL7', 'ENO1', 'FAM57A', 'INPP5A', 'LAYN', 'MDFI', 'MTP18', 'P4HA1', 'PDK1', 'POU6F2', 'PRDX4', 'PTHLH', 'SLC12A1', 'SOBP', 'TFAP2C', 'TMTC4', 'TNIP1', 'TPI1']
hypoxia_446     = ['ACTL8', 'AK3L1', 'EGLN3', 'GPR139', 'KISS1R', 'PFKL', 'PKM2', 'SLC16A3', 'SLC1A6', 'ADORA2A', 'BNIP3', 'C11orf88', 'C8orf58', 'CCDC19', 'CCDC64B', 'CYB5A', 'DGCR5', 'EGFL7', 'ENO1', 'FAM57A', 'INPP5A', 'LAYN', 'MDFI', 'MTP18', 'P4HA1', 'PDK1', 'POU6F2', 'PRDX4', 'PTHLH', 'SLC12A1', 'SOBP', 'TFAP2C', 'TMTC4', 'TNIP1', 'TPI1', 'TTYH3', 'TYRP1', 'ABP1', 'AGPAT2', 'AQP1', 'ATG9A', 'BCAT1', 'BNIP3L', 'C16orf82', 'C20orf200', 'C2orf54', 'C5orf62', 'C7orf68', 'C8orf22', 'CD300A', 'CLEC4C', 'COL17A1', 'DGCR10', 'DNAH11', 'EDARADD', 'FAM153C', 'FZD10', 'GABRD', 'GYS1', 'HK2', 'HTR5A', 'IMPDH1', 'ITPK1', 'NDRG1', 'NETO1', 'NTNG2', 'PCP4L1', 'PGAM1', 'PGK1', 'PLOD1', 'PTPRQ', 'RSPO1', 'SFXN3', 'SLC2A1', 'SLCO5A1', 'SNAPC1', 'UPB1', 'ABI3', 'ALDOA', 'BYSL', 'C10orf10', 'C1QTNF2', 'C22orf42', 'C4orf47', 'C5orf13', 'CA9', 'CABP1', 'CECR5', 'CLCNKA', 'CLCNKB', 'COX4I2', 'CTXN2', 'DDIT4L', 'DDX54', 'EEF2K', 'EML1', 'EPO', 'EXOC3L', 'FAM153B', 'FAM19A5', 'FAM9B', 'FGF11', 'GAPDH', 'GLI3', 'GMFG', 'GPC3', 'GPER', 'IL3RA', 'KRT84', 'LDHA', 'MARCKSL1', 'MFRP', 'MIER2', 'MIF', 'NECAB2', 'PDE4C', 'PDLIM2', 'PECAM1', 'R3HCC1', 'RAB40C', 'RIMKLA', 'SEC61G', 'SEMA3F', 'SPCS3', 'STC2', 'TBC1D26', 'TRAPPC9', 'VSIG1', 'ABCA4', 'AGRN', 'APOH', 'ARAF', 'ARHGDIB', 'ARMC4', 'BAX', 'BGN', 'BID', 'C13orf15', 'C19orf22', 'C9orf167', 'C9orf69', 'CD248', 'CD34', 'CERKL', 'CGNL1', 'CLEC14A', 'CX3CL1', 'DAD1L', 'DGCR9', 'ECSCR', 'EIF4EBP1', 'FAM26F', 'FKBP1A', 'FOLH1', 'GNB2', 'HLX', 'HNF1A', 'HSBP1', 'IFITM2', 'IGSF9', 'KDM4B', 'KIAA1751', 'LOXL3', 'LYPLA2', 'MCTP2', 'MMP14', 'MT3', 'MYL6', 'MYOM1', 'NDUFA4L2', 'NRBP1', 'ORAI1', 'PCSK6', 'PDAP1', 'PFKP', 'PFN1', 'PGF', 'PGM1', 'PTP4A3', 'RAP2B', 'RPLP0', 'SFRS9', 'SHMT2', 'SLC16A1', 'SLC22A23', 'SLC25A13', 'SLC28A1', 'SPARC', 'TARBP2', 'TMEM123', 'TMEM173', 'TMEM185A', 'TMEM204', 'TMEM45A', 'TSC22D1', 'UQCRH', 'VASP', 'VWA1', 'WDR54', 'ZNF395', 'A2M', 'A4GALT', 'ACPT', 'ACTN2', 'ADA', 'ADAMTS7', 'AK2', 'AMZ1', 'ANGPT2', 'ANKZF1', 'ANP32B', 'ANXA11', 'ARHGEF10', 'ARHGEF15', 'ARID5A', 'ARPC1B', 'ASB9', 'B4GALT2', 'BACE2', 'BATF3', 'BCKDK', 'BCL2L12', 'BMP1', 'C10orf105', 'C10orf47', 'C12orf27', 'C1QTNF1', 'C1orf126', 'C20orf166', 'C22orf45', 'C9orf135', 'CA2', 'CACNA1H', 'CALU', 'CAPZB', 'CAV2', 'CCDC102B', 'CD1D', 'CD79B', 'CDA', 'CDC42EP2', 'CDC42EP5', 'CDH13', 'CDH23', 'CENPB', 'CERK', 'CLEC11A', 'CLEC1A', 'COL18A1', 'COL27A1', 'COL4A1', 'COL4A2', 'COL5A3', 'COL6A2', 'COTL1', 'CPA3', 'CPM', 'CRYBB1', 'CSF1', 'CSK', 'CSPG4', 'CSRP2', 'CXorf36', 'DAZAP2', 'DDOST', 'DENND2A', 'DGKD', 'DLL4', 'DOCK6', 'DTX2', 'EDNRB', 'EEF2', 'EEFSEC', 'EFHD2', 'EFNA1', 'EGFR', 'EHD2', 'EIF4EBP2', 'ELOVL1', 'ENDOG', 'ENG', 'ERG', 'ERO1L', 'ESAM', 'FAM101B', 'FAM176B', 'FAM26E', 'FAM38A', 'FAM60A', 'FBL', 'FHL3', 'FJX1', 'FLJ11235', 'FLT1', 'FLT4', 'FOXC2', 'FOXD1', 'FOXL1', 'FSCN1', 'FZD4', 'GIMAP4', 'GIMAP5', 'GIPC3', 'GJA4', 'GJC1', 'GLT25D1', 'GNA14', 'GPI', 'GPR124', 'GPR160', 'GPR4', 'GPR56', 'GPRC5C', 'GPX8', 'GRAMD3', 'GRAP', 'HSPB7', 'ID3', 'IFITM3', 'IGFBP4', 'IGFBP7', 'IL4R', 'INSR', 'IPO4', 'IRGM', 'ITGA5', 'ITPRIP', 'JAG2', 'KBTBD11', 'KCNE3', 'KDM3A', 'KIAA0319L', 'KIAA2013', 'KRT18', 'LDB2', 'LEPRE1', 'LHFP', 'LINGO1', 'LMAN2', 'LMO4', 'LOC158376', 'LOXL2', 'LRRC32', 'LRRC70', 'LYPLA1', 'MAGED1', 'MFNG', 'MGLL', 'MLKL', 'MLYCD', 'MMP11', 'MRPS2', 'MYCT1', 'MYL12A', 'MYOF', 'NASP', 'NEDD9', 'NEUROD1', 'NFATC1', 'NID1', 'NOTCH4', 'NOX4', 'NRARP', 'NRAS', 'NUDT1', 'ODZ2', 'OLFML2A', 'OLFML2B', 'P2RY8', 'PCDH12', 'PDGFA', 'PDGFB', 'PES1', 'PIH1D1', 'PITPNC1', 'PLA2G4A', 'PLAC9', 'PLEKHG2', 'PLVAP', 'PLXND1', 'PPP1R13L', 'PPP4R1', 'PREX1', 'PSMB2', 'PTRF', 'RAB17', 'RAB32', 'RASGRP3', 'RASIP1', 'RBMS2', 'RCC2', 'RGS4', 'RILPL2', 'RIN3', 'ROBO4', 'RPL10', 'RPL11', 'RPS5', 'RPS9', 'SCARF1', 'SDF4', 'SEC14L1', 'SERBP1', 'SERPINH1', 'SGK223', 'SIPA1', 'SLC2A3', 'SMOC2', 'SMPDL3A', 'SMTN', 'SOX11', 'SPNS2', 'STAB1', 'STC1', 'STK40', 'SYDE1', 'SYNM', 'TBXA2R', 'TDO2', 'TEAD2', 'TEAD4', 'TFPI', 'TFR2', 'TGFB1', 'TGIF2', 'TIE1', 'TMC6', 'TMEM189', 'TMEM37', 'TNFAIP8L1', 'TOX2', 'TRIM28', 'TRPC4', 'TSKU', 'TSPAN14', 'TSPAN15', 'TUBB6', 'TXNDC12', 'UBAC2', 'UQCRHL', 'VEGFA', 'VWA3A', 'WDR1', 'WNT3', 'YBX1', 'ZYX']
sympathoadrenal = ['ASCL1', 'GATA3', 'TH', 'DBH', 'PNMT', 'CHGA', 'CHGB', 'HAND2', 'PHOX2B', 'GFAP', 'ENO2', 'STMN2', 'NPY', 'DDC', 'ADORA2A', 'DRD2', 'SYP', 'SYT4', 'TPH1']
neuralcrest     = ['NOTCH2', 'NES', 'GLIS3', 'IFFO1', 'SMAD3', 'SOX9', 'RUNX1', 'ID1', 'SNAI2', 'SOX9', 'VIM', 'GLI1', 'SF1', 'NT5E', 'ENG', 'CD44', 'MME', 'EGFR', 'FGFR1', 'WNT5A', 'PDGFRA', 'TGFBR2', 'FZD1', 'THBS1', 'ITGB3', 'NID2']

idx_var = [i for i, label in enumerate(data.var[1]) if label in list(set(hypoxia_446+sympathoadrenal+neuralcrest))]

print('selecting attributes ...')
data = data.select(idx_cols=idx_var)

# EXPORTAMOS A DICT QUE USA NUESTRA APP "interactive dual tSNE"!
data_object = data.export_to_dict()


Tm2 = 20


X = data_object['X']
M = data_object['M']
N = data_object['N']
var_labels = data_object['var'][1]
obs_labels = data_object['obs'][1]
can_labels = data_object['obs'][0]

selFil_idx = np.arange(M)
selCol_idx = np.arange(N)

idx_obs = np.arange(M)
idx_var = np.arange(N)



# muestras
x1 = X.copy()
y1 = np.random.randn(x1.shape[0],2)

# genes
x2 = X.T.copy()
y2 = np.random.randn(x2.shape[0],2)


# 2D map initialization
print('initializing maps (samples, genes) ...')
from umap import UMAP
umap = UMAP(random_state=1)
y1 = umap.fit_transform(X)
y2 = umap.fit_transform(X.T)



N = x1.shape[0]		# número de muestras
M = x2.shape[0]		# número de genes



# selection colors
colors_obs = ['#000000']*N
colors_var = ['#000000']*M


# column datas sources
source_obs = ColumnDataSource({'x1':y1[:,0],'y1':y1[:,1],'labels':obs_labels,'colors':colors_obs, 'can_labels':can_labels})
source_var = ColumnDataSource({'x2':y2[:,0],'y2':y2[:,1],'labels':var_labels,'colors':colors_var})

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
	source_var.data['y2'] = y2[:,1]

def reset_original_state():
	global y1,y2
	# train a UMAP
	print('training UMAP (samples and genes)...')
	y1 = umap.fit_transform(X)
	y2 = umap.fit_transform(X.T)

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
	selFil = [obs_labels[indice] for indice in source_obs.selected.indices]
	selFil_idx = source_obs.selected.indices
	textinput_colorsel_obs.value = ','.join([obs_labels[i] for i in selFil_idx])

def selection_callback_2Dcol(event):
	global selCol, selCol_idx
	selCol = [var_labels[indice] for indice in source_var.selected.indices]
	selCol_idx = source_var.selected.indices
	textinput_colorsel_var.value = ','.join([var_labels[i] for i in selCol_idx])

def update_conditional_dr():
	global idx_obs, idx_var
	global x1, x2

	idx_obs = get_idx_selection(textinput_colorsel_obs.value,obs_labels,no_empty=True)
	idx_var = get_idx_selection(textinput_colorsel_var.value,var_labels,no_empty=True)

	# show in textboxes the interpreted readings (for user check)
	textinput_colorsel_obs.value = ','.join([obs_labels[i] for i in idx_obs])
	textinput_colorsel_var.value = ','.join([var_labels[i] for i in idx_var])

	update_affinities()




# COLOR OF SAMPLES AND GENES
def update_color_obs_1():
	idx = get_idx_selection(textinput_colorsel_obs.value,obs_labels,no_empty=True)
	source_obs.data['colors'] = [textinput_color_obs.value if i in idx else source_obs.data['colors'][i] for i in range(N)]

def update_color_var_1():
	idx = get_idx_selection(textinput_colorsel_var.value,var_labels,no_empty=True)
	source_var.data['colors'] = [textinput_color_var.value if i in idx else source_var.data['colors'][i] for i in range(M)]

def update_color_obs(attr,old,new):
	idx = get_idx_selection(new,var_labels)
	
	# gene expression color scale
	if len(idx)==1:
		from matplotlib.cm import seismic
		from matplotlib.colors import rgb2hex
		textinput_color_obs.value = var_labels[idx[0]]
		source_obs.data['colors'] = [rgb2hex(seismic(np.interp(x,[-1,1],[0,1]))) for x in x1[:,idx[0]]]
	else:
		update_color_obs_1()
	print(attr,old,new)

def update_color_var(attr,old,new):
	idx = get_idx_selection(new,obs_labels)
	
	# gene expression color scale
	if len(idx)==1:
		from matplotlib.cm import seismic
		from matplotlib.colors import rgb2hex
		textinput_color_var.value = obs_labels[idx[0]]
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


curdoc().add_periodic_callback(update_tsne,Tm2)		# callback lenta

curdoc().add_root(layout(column(cabecera, row(
	column(fig1,slider_perplexity_1,slider_lr_1,textinput_color_obs,textinput_colorsel_obs),
	column(fig2,slider_perplexity_2,slider_lr_2,textinput_color_var,textinput_colorsel_var),
	column(button_conditional_dr,button_reset,button_obs,button_var,checkbox_group,button_exaggeration)
	))))

