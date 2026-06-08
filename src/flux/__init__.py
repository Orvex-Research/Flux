import sys
import transformers

# Inject transformers into sys.modules as flux so that all imports from flux point to transformers
sys.modules[__name__] = transformers
