from deepscratch.models.initializers.random_initialization import RandomInitialization
from deepscratch.models.initializers.he_et_al import HeEtAl

initializers = {
    'random': RandomInitialization,
    'he-et-al': HeEtAl
}


def load(initializer, **kwargs):
    if initializer not in initializers:
        raise ValueError('Unknown initializer: %s' % initializer)
    else:
        return initializers[initializer]