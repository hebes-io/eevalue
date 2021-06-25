import pandas as pd 


def get_sets(instance, varname):
    """Get sets that belong to a pyomo Variable or Param
    :param instance: Pyomo Instance
    :param varname: Name of the Pyomo Variable (string)
    :return: A list with the sets that belong to this Param
    """
    var = getattr(instance, varname)

    if var.dim() > 1:
        sets = [pset.getname() for pset in var._index.subsets()]
    else:
        sets = [var._index.name]
    return sets


def get_set_members(instance, sets):
    """Get set members relative to a list of sets
    :param instance: Pyomo Instance
    :param sets: List of strings with the set names
    :return: A list with the set members
    """
    sm = []
    for s in sets:
        sm.append([v for v in getattr(instance, s).data()])
    return sm


def pyomo_to_pandas(instance, varname, dates=None):
    """
    Function converting a pyomo variable or parameter into a pandas dataframe.
    The variable must have one or two dimensions and the sets must be provided as a list of lists
    :param instance: Pyomo model instance
    :param varname: Name of the Pyomo Variable (string)
    :param dates: List of datetimes or pandas DatetimeIndex
    """
    setnames = get_sets(instance, varname)
    sets = get_set_members(instance, setnames)
    var = getattr(instance, varname)  # Previous script used model.var instead of var
    ####
    if len(sets) != var.dim():
        raise ValueError('The number of provided set lists (' + str(
            len(sets)) + ') does not match the dimensions of the variable (' + str(var.dim()) + ')')

    if var.dim() == 1:
        [SecondSet] = sets
        out = pd.DataFrame(columns=[var.name], index=SecondSet)
        data = var.get_values()
        for idx in data:
            out[var.name][idx] = data[idx]
        
        if dates is not None:
            out = out.set_index(dates)

    elif var.dim() == 2:
        [FirstSet, SecondSet] = sets
        out = pd.DataFrame(columns=FirstSet, index=SecondSet)
        data = var.get_values()
        for idx in data:
            out[idx[0]][idx[1]] = data[idx]
        
        if dates is not None:
            out = out.set_index(dates)
            
    else:
        raise ValueError('the pyomo_to_pandas function only accepts one or two-dimensional variables')

    return out