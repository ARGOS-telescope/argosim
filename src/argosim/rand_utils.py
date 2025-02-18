"""Random utils.

A small module to hold a class for a temporary seed,
so as to not mess with any global seed.

:Authors: Samuel Gullin <gullin@ia.forth.gr>

"""

import numpy as np


class local_seed:
    """
    Holds a seed for use with `numpy.random` in a local scope.

    Can be instantiated with no seed, in which case it will yield the default behaviour.
    Note that if `seed` is set to the same seed `numpy` was using outside this scope,
    that doesn't mean the state is the same -- if you intend to keep the same state,
    use `seed = None`.

    Attributes
    ----------
    seed: int, optional
        The seed used for numpy.random in the local scope.
    old_state:
        The state of numpy.random outside this scope, to be resumed afterwards.
    """

    def __init__(self, seed=None):
        """Instantiate with a set seed."""
        self.seed = seed
        self.old_state = None

    def __enter__(self):
        """
        Enter local scope.

        Grabs `numpy.randoms`'s state only if a `seed != None` was supplied
        upon creation, and then seeds `numpy.random`.
        """
        if self.seed:
            self.old_state = np.random.get_state()
            np.random.seed(self.seed)

    def __exit__(self, *_):
        """
        Exit local scope.

        Reverts `numpy.random` to the outer scope's state, if a `seed != None`
        was supplied upon creation.
        """
        if self.seed:
            np.random.set_state(self.old_state)
