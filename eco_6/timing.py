"""
import with
from timing import timing, timing_finalize
--

1.
decorate each timed function with
@timing

2.
after all functions timed, output results with
timing_finalize()
outputs with function name, and accumulated time taken
"""
from time import process_time
from functools import wraps

trackingObj = {} # note that any script that uses the same script will all insert into the exact same dict
# such as evo and driver both inserting into this obj, despite only having the import as the commonality

def timing(func):
    @wraps(func) # helps keep docstrings the same
    def wrapper(self, *args, **kwargs):
        # start
        start = process_time()
        
        # call func
        result = func(self, *args, **kwargs)
        
        # logging
        taken = process_time() - start
        try:
            # existing key
            trackingObj[func.__name__] += taken
        except KeyError:
            # non existent key
            trackingObj[func.__name__] = 0
            trackingObj[func.__name__] += taken

        return result
    return wrapper

def getTimeTrackedObjs():
    return trackingObj