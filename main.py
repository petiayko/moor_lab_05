from stochastic_filter import StochasticFilter

# точка входа в программу
if __name__ == '__main__':
    task_three = StochasticFilter(r=3)
    task_three.graphic()

    task_five = StochasticFilter(r=5)
    task_five.graphic()
