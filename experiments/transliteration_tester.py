import subprocess, sys, os
sys.path.append(r'../indic-trans')
from indictrans import Transliterator
trn = Transliterator(source='eng', target='hin', build_lookup=True)
result = trn.transform("hello")
print(result)
