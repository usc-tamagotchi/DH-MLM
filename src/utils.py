import os


def get_ds_name_from_path(path_ds):
    if path_ds[-1] == '/':
        path_ds = path_ds[:-1]

    path_ds = os.path.basename(path_ds)
    ds_name = os.path.basename(path_ds).replace('.ds', '')

    return ds_name


def get_model_name_from_path(path_ds, include_ckpt=False):
    if path_ds[-1] == '/':
        path_ds = path_ds[:-1]

    basename = os.path.basename(path_ds)
    if 'checkpoint' in basename:
        model_name = os.path.basename(os.path.dirname(path_ds))
        if not include_ckpt:
            return model_name
        else:
            step = basename.split('-')[1]
            return f'{model_name}-ckpt-{step}'
    else:
        return basename
