from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# декоратор для выполнения функции в отдельном потоке
def separate_thread(function):
    def wrapper(*arg):
        return Thread(name=function.__name__, target=function, args=(*arg,)).start()

    return wrapper


class StochasticFilter:
    # заданные константы
    x_min = 0
    x_max = np.pi
    K = 100
    P = 0.95
    E = 0.01
    L = 10
    a = 0.25

    def __init__(self, r):
        # размер скользящего окна
        self.__r = r
        print(f'r={self.__r}')

        self.__x_k = [self.x_min + (self.x_max - self.x_min) * k / self.K for k in range(self.K)]
        self.__noise = [self.__func_noise(x) for x in self.__x_k]
        self.__M = round((self.__r - 1) / 2)
        self.__N = round(np.log(1 - self.P) / np.log(1 - (self.E / (self.x_max - self.x_min))))
        self.__array_lambda = [i / self.L for i in range(self.L + 1)]
        self.__filtered = self.__filter()

    # заданная функция
    @staticmethod
    def __func(x):
        return np.sin(x) + 0.5

    # метод получение шума функции
    def __func_noise(self, x):
        return StochasticFilter.__func(x) + np.random.uniform(-self.a, self.a)

    # метод создания датафрейма, его сохранения - в отдельном потоке
    @separate_thread
    def __save_table(self, data):
        table_values = pd.DataFrame(data=data)
        table_values.to_excel(f'table_for_r{self.__r}.xlsx', index=False)

    # метод отображения графика функции на экране
    def graphic(self):
        plt.plot(self.__x_k, [self.__func(x) for x in self.__x_k], label='f(x) = sin(x) + 0.5')
        plt.plot(self.__x_k, self.__noise, label='noise')
        plt.plot(self.__x_k, self.__filtered, label='filtered')
        plt.title(f'Function (r={self.__r})')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid()
        plt.show()

    # метод получения вектора альфа
    def __get_alpha(self):
        alpha = np.zeros(self.__r)
        alpha[self.__M] = np.random.uniform(0, 1)
        for i in range(1, self.__M):
            alpha[self.__M - i] = alpha[self.__M + i] = 0.5 * np.random.uniform(0, 1 - sum(alpha))
        alpha[0] = alpha[self.__r - 1] = 0.5 * (1 - sum(alpha))
        return alpha

    # метод расчета взвешенного скользящего среднего (средним гармоническим)
    def __harmonic_mean(self, k, alpha):
        mean = 0.
        if k - self.__M < 0:
            for i in range(k + self.__M + 1):
                mean += alpha[i] / self.__noise[i]
        elif k + self.__M > self.K - 1:
            for i in range(k - self.__M, self.K):
                mean += alpha[i - k + self.__M] / self.__noise[i]
        else:
            for i in range(k - self.__M, k + self.__M + 1):
                mean += alpha[i - k + self.__M] / self.__noise[i]
        return 1 / mean

    # метод расчета уровня зашумленности
    def __get_noise_criterion(self, alpha):
        omega = 0
        for k in range(1, self.K + 1):
            curr = max(self.__harmonic_mean(k, alpha), self.__harmonic_mean(k - 1, alpha))
            if curr > omega:
                omega = curr
        return omega

    # метод получения критерия отличия
    def __get_difference_criterion(self, alpha):
        maximum = 0
        for k in range(self.K + 1):
            curr = abs(self.__harmonic_mean(k, alpha) - self.__func_noise(k))
            if curr > maximum:
                maximum = curr
        return maximum

    # метод получения интегрального критерия
    def __get_j(self, lambda_, alpha):
        return lambda_ * self.__get_noise_criterion(alpha) + (1 - lambda_) * self.__get_difference_criterion(alpha)

    # метод получения оптимального альфа
    def __get_best_alpha(self, lambda_):
        best_alpha = self.__get_alpha()
        min_j = self.__get_j(lambda_, best_alpha)
        for i in range(self.__N - 1):
            cur_alpha = self.__get_alpha()
            cur_j = self.__get_j(lambda_, cur_alpha)
            if cur_j < min_j:
                min_j = cur_j
                best_alpha = cur_alpha
        return best_alpha

    # метод получения расстояния Чебышева
    def __get_distance(self, alpha):
        return max((self.__get_difference_criterion(alpha)), (self.__get_noise_criterion(alpha)))

    # метод очищения сигнала
    def __filter(self):
        arr_distance = []
        arr_alpha = []
        arr_w = []
        arr_d = []

        min_dist_j = None
        best_alpha = None
        best_lambda = None
        for lambda_ in self.__array_lambda:
            alpha = self.__get_best_alpha(lambda_)
            cur_dist_j = self.__get_distance(alpha)
            arr_alpha.append(alpha)
            arr_distance.append(cur_dist_j)
            arr_w.append(self.__get_noise_criterion(alpha))
            arr_d.append(self.__get_difference_criterion(alpha))
            if not min_dist_j or cur_dist_j < min_dist_j:
                min_dist_j = cur_dist_j
                best_alpha = alpha
                best_lambda = lambda_

        self.__save_table({
            'h': self.__array_lambda,
            'dist': arr_distance,
            'alpha': arr_alpha,
            'w': arr_w,
            'd': arr_d
        })

        plt.plot(arr_w, arr_d, 'o')
        plt.grid()
        plt.show()

        filtered = [self.__harmonic_mean(k, best_alpha) for k in range(self.K)]
        print(f'h={best_lambda}')
        print(f'J={self.__get_j(best_lambda, best_alpha)}\n')
        return filtered
