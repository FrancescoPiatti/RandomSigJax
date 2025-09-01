from numbers import Number

def _check_positive_value(scalar: Number, name: str) -> Number:
  """
  Checks whether `scalar` is a positive number.
  """
  if scalar <= 0:
    raise ValueError(f'The parameter \'{name}\' should be positive.')
  

def _check_positive_integer_value(scalar: Number, name: str) -> Number:
  """
  Checks whether `scalar` is a positive integer.
  """
  if not isinstance(scalar, int) or scalar <= 0:
    raise ValueError(f'The parameter \'{name}\' should be a positive integer.')
  

def _check_non_negative_value(scalar: Number, name: str) -> Number:
  """
  Checks whether `scalar` is a positive number.
  """
  if scalar < 0:
    raise ValueError(f'The parameter \'{name}\' should be non negative.')


def _check_boolean(boolean: bool, name: str) -> bool:
  """
  Checks whether `boolean` is a boolean.
  """
  if not isinstance(boolean, bool):
    raise ValueError(f'The parameter \'{name}\' should be a boolean.')


def _check_string(string: str, name: str) -> str:
  """
  Checks whether `string` is a string.
  """
  if not isinstance(string, str):
    raise ValueError(f'The parameter \'{name}\' should be a string.')