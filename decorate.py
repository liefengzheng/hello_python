from functools import wraps

def a_new_decorator(a_func):
  @wraps(a_func)
  def wrapTheFunction(*args, **kwargs):
    print("I am doing some boring work before executing a_func()")
    a_func(*args, **kwargs)
    print("I am doing some boring work after executing a_func()")
  return wrapTheFunction  

@a_new_decorator
def a_function_requiring_decoration(name = 'hello'):
    """Hey yo! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")
    print(name + ": liefeng" )

print(a_function_requiring_decoration.__name__)  
a_function_requiring_decoration("haha")