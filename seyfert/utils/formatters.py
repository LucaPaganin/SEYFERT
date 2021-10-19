from typing import Tuple
import datetime


def str_to_bool(s: "str") -> "bool":
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def str_to_float(string):
    try:
        return float(string)
    except ValueError:
        return eval(string)


def time_format(dt: "float") -> "Tuple":
    """!
    Function to format a time interval from seconds to hh:mm:ss
    The function returns a tuple of integers: (h, m, s)
    :param dt: A time interval in seconds
    """
    hours, remainder = divmod(int(dt), 3600)
    mins, secs = divmod(remainder, 60)
    millisecs = round(1000*(dt - int(dt)), 3)
    return hours, mins, secs, millisecs


# noinspection PyTupleAssignmentBalance
def string_time_format(dt: "float"):
    h, m, s, ms = time_format(dt)
    string_formatted_time = f'{h}h {m}m {s}s {ms}ms'
    return string_formatted_time


def datetime_str_format(t: "datetime.datetime", hour_sep: "str" = "-") -> "str":
    return t.strftime(f'%Y-%m-%dT%H{hour_sep}%M{hour_sep}%S')


def date_from_str(date_string: "str", hour_sep: "str" = "-") -> "datetime.datetime":
    return datetime.datetime.strptime(date_string, f'%Y-%m-%dT%H{hour_sep}%M{hour_sep}%S')
