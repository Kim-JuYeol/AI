x = 10
learning_rate = 0.1
max_iterations = 1000

loss_func = lambda x: (x-3)**2 + 10

gradient = lambda x: 2*x - 6

for i in range(max_iterations):
    x = x - learning_rate * gradient(x)
    print("손실함수값(", x, ")=", loss_func(x))
    
print("최소값 = ", x)