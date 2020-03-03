# Replay winning individual
def evaluate_best(calculator, best, seed, num_trials=100):
    if best:
        test_scores = []
        avg_score = 0

        for i in range(num_trials):
            render_flag = False
            # if i == 0:
            #    render_flag = True
            test_scores += [calculator.calculate_fitness(best, render=render_flag)]
            seed += 1

        avg_score = sum(test_scores) / len(test_scores)
        print(f"The best individual scored {avg_score} on average")