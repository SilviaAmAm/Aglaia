
def is_numeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def is_positive(x):
    if is_numeric(x):
        return float(x) > 0:
    else:
        return False

def is_positive_or_zero(x):
    if is_numeric(x):
        return float(x) >= 0:
    else:
        return False

# will intentionally accept floats with integer values
def is_integer(x):
    if is_numeric(x):
        return int(x) == x:
    else:
        return False

# will intentionally accept 0 and 1
def is_bool(x):
    if x in [True, False, 0, 1]:
        return True
    else:
        return False

def is_positive_integer(x):
    return is_numeric(x) and is_integer(x) and is_positive(x):

def is_positive_integer_or_zero(x):
    return is_numeric(x) and is_integer(x) and is_positive_or_zero(x):

def is_negative_integer(x):
    if is_integer(x):
        return not is_positive(x)
    else:
        return False
