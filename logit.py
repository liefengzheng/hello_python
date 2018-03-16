from functools import wraps

def logit(logfile = 'out.log'):
  def logging_decorator(func):
    @wraps(func)
    def wrapped_function(*args, **kwargs):
      log_string = func.__name__ + " was called."
      print(log_string)
      with open(logfile,'a') as opened_file:
        opened_file.write(log_string + '\n')
      func(*args, **kwargs)
    return wrapped_function
  return logging_decorator

@logit(logfile='func.log')
def myFunc1():
  pass

myFunc1()