import importlib.resources

def get_module_res(module, name):
    return importlib.resources.files(module).joinpath(name).open("rb")
