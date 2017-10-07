import numpy as np
import hashlib
import operator
import pickle
import os

def get_default_args(func):
    u"""
    returns a dictionary of arg_name:default_values for the input function
    """
    import inspect
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults:
        return dict(list(zip(args[-len(defaults):], defaults)))
    else:
        return {}

def hash_obj(obj):
    u"""
    return a hash of any python object
    """
    from meta.decompiler import decompile_func
    import ast
    def obj_digest(to_digest):
        return hashlib.sha1(pickle.dumps(to_digest)).hexdigest()

    def iter_digest(to_digest):
        return obj_digest(reduce(operator.add, list(map(hash_obj, to_digest))))

    if (not isinstance(obj, np.ndarray)) and hasattr(obj, u'__iter__') and (len(obj) > 1):
        if isinstance(obj, dict):
            return iter_digest(iter(obj.items()))
        else:
            return iter_digest(obj)
    else:
        # Functions receive special treatment, such that code changes alter
        # the hash value
        if hasattr(obj, u'__call__'):
            try:
                return obj_digest(ast.dump(decompile_func(obj)))
            # This covers an exception that happens in meta.decompiler under
            # certain situations. TODO: replace this workaround with something
            # better.
            except IndexError:
                return obj_digest(pickle.dumps(obj))
        else:
            return obj_digest(obj)

def hashable_dict(d):
    u"""
    try to make a dict convertible into a frozen set by 
    replacing any values that aren't hashable but support the 
    python buffer protocol by their sha1 hashes
    """
    #TODO: replace type check by check for object's bufferability
    for k, v in d.items():
        # for some reason ndarray.__hash__ is defined but is None! very strange
        #if (not isinstance(v, collections.Hashable)) or (not v.__hash__):
        if isinstance(v, np.ndarray):
            d[k] = hash_obj(v)
    return d

def eager_persist_to_file(file_name, excluded = None, rootonly = True):
    u"""
    Decorator for memoizing function calls to disk.
    Differs from persist_to_file in that the cache file is accessed and updated
    at every call, and that each call is cached in a separate file. This allows
    parallelization without problems of concurrency of the memoization cache,
    provided that the decorated function is expensive enough that the
    additional read/write operations have a negligible impact on performance.
    Inputs:
        file_name: File name prefix for the cache file(s)
        rootonly : boolean
                If true, caching is only applied for the MPI process of rank 0.
    """
    cache = {}

    def decorator(func):
        #check if function is a closure and if so construct a dict of its bindings
        if func.func_code.co_freevars:
            closure_dict = hashable_dict(dict(list(zip(func.func_code.co_freevars, (c.cell_contents for c in func.func_closure)))))
        else:
            closure_dict = {}

        def gen_key(*args, **kwargs):
            u"""
            Based on args and kwargs of a function, as well as the 
            closure bindings, generate a cache lookup key
            """
            # union of default bindings in func and the kwarg bindings in new_func
            # TODO: merged_dict: why aren't changes in kwargs reflected in it?
            merged_dict = get_default_args(func)
            if not merged_dict:
                merged_dict = kwargs
            else:
                for k, v in merged_dict.items():
                    if k in kwargs:
                        merged_dict[k] = kwargs[k]
            if excluded:
                for k in list(merged_dict.keys()):
                    if k in excluded:
                        merged_dict.pop(k)
            key = hash_obj(tuple(map(hash_obj, [args, merged_dict, list(closure_dict.items()), list(kwargs.items())])))
            #print "key is", key
#            for k, v in kwargs.iteritems():
#(                print k, v)
            return key

        @ifroot# TODO: fix this
        def dump_to_file(d, file_name):
            os.system(u'mkdir -p ' + os.path.dirname(file_name))
            with open(file_name, 'wb') as f:
                pickle.dump(d, f)
            #print "Dumped cache to file"
    
        def compute(*args, **kwargs):
            file_name = kwargs.pop(u'file_name', None)
            key = gen_key(*args, **kwargs)
            value = func(*args, **kwargs)
            cache[key] = value
            # Write to disk if the cache file doesn't already exist
            if not os.path.isfile(file_name):
                dump_to_file(value, file_name)
            return value

        def new_func(*args, **kwargs):
            # Because we're splitting into multiple files, we can't retrieve the
            # cache until here
            #print "entering ", func.func_name
            key = gen_key(*args, **kwargs)
            full_name = file_name + key
            if key not in cache:
                try:
                    try:
                        with open(full_name, 'rb') as f:
                            cache[key] = pickle.load(f)
                    except EOFError:
                        os.remove(full_name)
                        raise ValueError(u"Corrupt file")
                    #print "cache found"
                except (IOError, ValueError):
                    #print "no cache found; computing"
                    compute(*args, file_name = full_name, **kwargs)
            # if the "flush" kwarg is passed, recompute regardless of whether
            # the result is cached
            if u"flush" in list(kwargs.keys()):
                kwargs.pop(u"flush", None)
                # TODO: refactor
                compute(*args, file_name = full_name, **kwargs)
            #print "returning from ", func.func_name
            return cache[key]

        return new_func

    return decorator
