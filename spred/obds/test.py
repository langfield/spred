num_workers = 52
sec_per_file = 360
tokens = [0] * sec_per_file
iterations = sec_per_file // (num_workers - 1)
remainder = sec_per_file % (num_workers - 1)

# Processes.
for i in range(num_workers):
    steps = iterations * (num_workers - 1)

    if i == num_workers - 1:
        iterations = remainder

    print("Steps:", steps)
    print("Iterations:", iterations)
    for j in range(iterations):
        if tokens == []:
            raise ValueError("Oh no!")
        tokens.pop()
        print(tokens)
    print("=============")
