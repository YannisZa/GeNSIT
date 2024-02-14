import click
from copy import deepcopy

from gensit.static.global_variables import *
from gensit.utils.misc_utils import unstringify
def btt_callback(ctx, param, value):
    if value is None:
        return {}
    else:
        value = list(value)

    # Convert burnin, thinning, trimming to dictionary
    if len(value) > 0:
        burnin_thinning_trimming = []
        for v in value:
            burnin_thinning_trimming.append({
                v[0]:{"burnin":v[1],"thinning":v[2],"trimming":v[3]}
            })
    else:
        burnin_thinning_trimming = []
    return burnin_thinning_trimming


def list_of_lists(ctx, param, value):
    if value is None:
        return []
    elif len(value) == 0:
        return []
    else:
        return [list(v) for v in value]
    
def evaluate_kwargs_callback(ctx, param, value):
    if value is None:
        return {}
    kwargs = []
    for keyval in value:
        if '=' in keyval:
            # Split on first occurence of = sign
            key,val = keyval.split('=',1)
            kwargs.append((key.strip(' '),val.strip(' ')))
        else:
            kwargs.append((keyval.strip(' '),keyval.strip(' ')))
    return kwargs

def unstringify_callback(ctx, param, value):
    if not value:
        return [{}]
    res = []
    for val in list(value):
        # Unstringify argument
        res.append(unstringify(val))
    return res
    
def list_of_str(ctx, param, value):
    if value is None:
        return []
    elif len(value) == 0:
        return []
    else:
        return list(map(str,list(value)))
    

def to_list(ctx, param, value, nargs:int=None):
    if value is None:
        return None
    elif len(value) == 0:
        return []
    else:
        return list(value)[:nargs]

def to_list2d(ctx, param, value):
    if value is None:
        return None
    elif len(value) == 0:
        return []
    else:
        return [list(value)]

def split_to_list(ctx, param, value, nargs:int=None):
    if value is None:
        return None
    else:
        return [list(v.split("&")) for v in list(value) if v is not None][:nargs]

def coordinate_parse(ctx, param, value):
    if value is None:
        return None
    else:
        res = [[],[]]
        value = list(value)
        for i in range(2):
            if i < len(value):
                if not value[i]:
                    res[i] = []
                else:
                    res[i] = list(value[i].split("&"))
            else:
                res[i] = []
        return res
    
def dims_to_list(ctx, param, value):
    if value is None:
        return None
    else:
        res = []
        for item in value:
            for v in list(item[1:]):
                tup = []
                if v is None or v == '':
                    continue
                tup.append(list(v.split("&")))
            res.append((item[0],*tup)) 
    return res

def coordinate_slice_callback(ctx, param, value):
    result = []
    for v in value:
        result.append([v[0],v[1],v[2].split(',')])
    return result

def unpack_statistics(ctx, param, value):
    # Unpack statistics if they exist
    statistics = {}
    for metric,stat,dim in value:
        # print('stat',stat)
        # print('dim',dim)
        if isinstance(stat,str):
            if '&' in stat:
                stat_unpacked = stat.split('&')
            else:
                stat_unpacked = [stat]
        elif hasattr(stat,'__len__'):
            stat_unpacked = deepcopy(stat)
        else:
            raise Exception(f'Statistic name {stat} of type {type(stat)} not recognized')
        if isinstance(dim,str): 
            if '&' in dim:
                dim_unpacked = dim.split('&')
            else:
                dim_unpacked = [str(dim)]
        elif hasattr(dim,'__len__'):
            dim_unpacked = list(map(str,dim))
        else:
            raise Exception(f'Statistic dim {dim_unpacked} of type {type(dim_unpacked)} not recognized')
        # Make sure number of statistics and axes provided is the same
        assert len(stat_unpacked) == len(dim_unpacked)
        substatistics = []
        for substat,subdim in list(zip(stat_unpacked,dim_unpacked)):
            # print('substat',substat)
            # print('subdim',subdim)
            substat_unpacked = [_s for _s in substat.split('|')] if '|' in substat else [substat]
            subdim_unpacked = [_dim for _dim in subdim.split('|')] if '|' in subdim else [subdim]
            # Unpack individual axes
            subdim_unpacked = [[str(_subdim) for _subdim in _dim.split("+")] \
                        if (_dim is not None and len(_dim) > 0) \
                        else None
                        for _dim in subdim_unpacked]
            # print('substat_unpacked',substat_unpacked)
            # print('subdim_unpacked',subdim_unpacked)
            # Add statistic name and axes pair to list
            substatistics.append(list(zip(substat_unpacked,subdim_unpacked)))
        
        # Add statistic name and axes pair to list
        if metric in statistics:
            statistics[metric].append(substatistics)
        else:
            statistics[metric] = substatistics
    return statistics


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        if (value is None) | (value == '[]'):
            return
        else:
            try:
                res = []
                for item in value:
                    res.append(ast.literal_eval(item))
                return res
            except ValueError:
                raise click.BadParameter(value)
    
class NotRequiredIf(click.Option):
    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop('not_required_if')
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually exclusive with %s' %
            self.not_required_if
        ).strip()
        super(NotRequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        we_are_present = self.name in opts
        other_present = self.not_required_if in opts

        if other_present:
            if we_are_present:
                raise click.UsageError(
                    "Illegal usage: `%s` is mutually exclusive with `%s`" % (
                        self.name, self.not_required_if))
            else:
                self.prompt = None

        return super(NotRequiredIf, self).handle_parse_result(
            ctx, opts, args)

class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval