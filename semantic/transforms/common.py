

class Compose(object):
	"""Composes several transforms together.
	Single arg -> can be used for transforms with a single arg
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, sample):
		for t in self.transforms:
			if t is not None:
				sample = t(sample)
		return sample