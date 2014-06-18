from pystruct.models import GridCRF, GraphCRF
class WeightedGridCRF(GridCRF):
	""" Variant of pystruct's Grid CRF that has scaling turned on """
	def __init__(self, n_states=None, n_features=None, inference_method=None,
			 neighborhood=4, class_weight=None):
		self.neighborhood = neighborhood
		GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
			inference_method=inference_method, 
			class_weight=class_weight)
