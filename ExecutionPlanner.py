from functools import partial, wraps
from typing import Callable, Hashable, List
from decohints import decohints



class ExecutionPlanner:
    def __init__(self, stacks_keys=None):
        self.stacks_keys: List = stacks_keys
        if stacks_keys is None:
            self.stacks_keys = [1]

        self.stacks: dict = {}
        for stack_key in self.stacks_keys:
            self.stacks[stack_key] = list()

    def add_function(self, function: Callable, stack_key: Hashable = 1):
        self.stacks[stack_key].append(function)

    def execute(self, arg, specified_stacks=None, ret_raw=True):
        results = {}
        if not specified_stacks:
            specified_stacks = [1]
        for stack_key in specified_stacks:
            result = arg
            for func in self.stacks[stack_key]:
                result = func(result)
            results[stack_key] = result
        if len(specified_stacks) == 1 and ret_raw:
            return results[specified_stacks[0]]
        else:
            return results


@decohints
def partial_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        return partial_func
    return wrapper
