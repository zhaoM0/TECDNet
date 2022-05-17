import importlib 
import os 
from os import path as osp 

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
	osp.splitext(osp.basename(v))[0] for v in os.listdir(arch_folder)
	if v.endswith('_arch.py')
]
# import all the arch modules
_arch_modules = [
	importlib.import_module(f'models.{file_name}')
	for file_name in arch_filenames
]

def dynamic_instantiation(modules, cls_type, opt):
	""" Dynamically instantiate class 

	Args:
		modules (list[importlib modules]): List of modules from importlib files.
		cls_type (str): class type.
		opt (dict): Class initizlization kwargs.

	Returns:
		class: Instantiated class.
	"""
	for module in modules:
		cls_ = getattr(module, cls_type, None)
		if cls_ is not None:
			break
	if cls_ is None:
		raise ValueError(f'{cls_type} is not found.')
	return cls_(**opt)

def define_network(cls_type, opt):
	""" define network module

	Args:
		cls_type (str): 

	"""
	net = dynamic_instantiation(_arch_modules, cls_type, opt)
	return net 
