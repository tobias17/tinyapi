from tinygrad import Tensor, TinyJit

@TinyJit
def f() -> Tensor:
   return Tensor.rand(1).realize()

for i in range(8):
   print(f().item())





# from tinygrad import Tensor, TinyJit

# @TinyJit
# def f(x:Tensor) -> Tensor:
#    return x.softmax().multinomial()

# inputs = [Tensor.rand(100) for _ in range(2)]

# for i in range(4):
#    print(f"Iteration {i+1}")
#    for inp in inputs:
#       print(f(inp.realize()).item())
#    print()


