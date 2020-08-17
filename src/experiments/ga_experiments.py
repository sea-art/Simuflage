"""
This file contains all experiments regarding the GA itself.
"""

import time
import multiprocessing


from DSE.exploration.GA.algorithm import initialize_sesp, GA


def run_ga(pop_size, nr_gens, nr_samples, sesp, eval_method, file_name):
    ga = GA(pop_size, nr_gens, nr_samples, sesp, eval_method=eval_method, log_filename=file_name)

    time1 = time.time()
    ga.run()
    time2 = time.time()

    print("Execution time " + eval_method + ": ", time2 - time1)


def changes_over_generations(pop_size, nr_gens):
    sesp = initialize_sesp()

    jobs = list()
    jobs.append(multiprocessing.Process(target=run_ga,
                                        args=(pop_size, nr_gens, 10 * pop_size, sesp, 'mcs', "out/mcs.csv")))
    jobs.append(multiprocessing.Process(target=run_ga,
                                        args=(pop_size, nr_gens, 10 * pop_size, sesp, 'ssar', "out/ssar.csv")))
    jobs.append(multiprocessing.Process(target=run_ga,
                                        args=(pop_size, nr_gens, 10 * pop_size, sesp, 'pucb', "out/pucb.csv")))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


if __name__ == "__main__":
    changes_over_generations(100, 70)