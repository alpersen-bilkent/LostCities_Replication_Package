# Some Windows environments end up with two different packages each
# bundling their own copy of Intel's OpenMP runtime (libiomp5md.dll) --
# in this project's case, almost certainly PyTorch's own bundled copy
# colliding with a separately-linked MKL/OpenMP copy pulled in by
# numpy/scipy or a geospatial dependency. The duplicate-initialization
# race can corrupt process-wide native state badly enough to crash
# unrelated later code (observed here: matplotlib's Agg renderer
# aborting deep inside a figure save, nothing to do with OpenMP itself).
# KMP_DUPLICATE_LIB_OK=TRUE is Intel's own documented workaround; setting
# it here (before any submodule -- and therefore before numpy/torch/
# matplotlib -- gets imported) makes every script in this project safe
# by default, with no manual environment-variable step required by
# anyone who clones/runs this code.
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
