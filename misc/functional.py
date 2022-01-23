def flatten(the_lists):
    result = []
    for item in the_lists:
        if isinstance(item, list):
            result += item
        else:
            result.append(item)
    if any(isinstance(item, list) for item in result):
        result = flatten(result)
    return result


def apply(item, fun):
    if isinstance(item, list):
        return [apply(x, fun) for x in item]
    else:
        return fun(item)
