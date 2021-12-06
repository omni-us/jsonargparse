import os

if 'JSONARGPARSE_OMEGACONF_FULL_TEST' in os.environ:
    import warnings
    from jsonargparse.loaders_dumpers import loaders
    if 'omegaconf' in loaders:
        loaders['yaml'] = loaders['omegaconf']
        warnings.warn('Running all tests with omegaconf as the yaml loader.')
