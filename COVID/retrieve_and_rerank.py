

from .retrieve_docs import get_first_n_1, pprint

docs = (
    get_first_n_1(
        qtext       = 'A pneumonia outbreak associated with a new coronavirus of probable bat origin',
        n           = 100,
        max_year    = 2021
    )
)


