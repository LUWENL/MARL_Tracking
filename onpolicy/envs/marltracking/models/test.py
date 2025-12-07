import numpy as np
from task_allocation import ga_plain, ga_kernel
from run_SAC import METADATA

if __name__ == '__main__':
    N_sat = 20
    N_tar = 40
    seed = 1234
    is_parallel = METADATA['seed']

    np.random.seed(seed)

    sat_samplings = np.random.rand(N_sat)  # np.array (N_sat)
    sat_attitudes = np.random.rand(N_sat, N_tar)  # np.array (N_sat, N_tar) q^e_4
    sat_vectors = np.random.rand(N_sat, 2)  # np.array (N_sat, 2)  trajectory vector
    sat_occultated = np.random.choice([True, False], size=(N_sat, N_tar))  # np.array (N_sat, N_tar) Boolean value

    tar_prioritys = np.random.rand(N_tar)  # np.array (N_tar)
    tar_vectors = np.random.rand(N_tar, 2)  # np.array (N_tar, 2)

    accumulative_time = 0

    for i in range(50):
        if is_parallel:
            best_allocation, best_fitness, consumed_time = ga_kernel(N_sat, N_tar, sat_samplings, sat_attitudes,
                                                                     sat_vectors, sat_occultated, tar_prioritys,
                                                                     tar_vectors)
        else:
            best_allocation, best_fitness, consumed_time = ga_plain(N_sat, N_tar, sat_samplings, sat_attitudes,
                                                                    sat_vectors, sat_occultated, tar_prioritys,
                                                                    tar_vectors)

        print("Task Allocation : {}".format(best_allocation))
        print("Best Fitness : {}".format(best_fitness))
        print("Consumed Time : {}".format(consumed_time))
        print("==" * 5)

        accumulative_time += consumed_time

    print(accumulative_time / 10)
