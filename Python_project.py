import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import date
import itertools
import math
import yfinance
from typing import List
import time

class PortfolioBuilder:

    def get_daily_data(self, tickers_list: List[str], start_date: date, end_date: date = date.today()) -> pd.DataFrame:
        """
        get stock tickers adj_close price for specified dates.
        :param List[str] tickers_list: stock tickers names as a list of strings.
        :param date start_date: first date for query
        :param date end_date: optional, last date for query, if not used assumes today
        :return: daily adjusted close price data as a pandas DataFrame
        :rtype: pd.DataFrame
        example call: get_daily_data(['GOOG', 'INTC', 'MSFT', ''AAPL'], date(2018, 12, 31), date(2019, 12, 31))
        """
        try:
            self.company = len(tickers_list)
            self.data = web.DataReader(tickers_list, data_source='yahoo', start=start_date, end=end_date)
            self.data = self.data.loc[:, 'Adj Close']
            self.days = self.data.shape[0]                                 # it's return tuple row and col 0 - Takes the row
            self.Stock_prices = self.data.to_numpy()                       # To reformat from panda to numpy list
            if self.data.isnull().values.any():
                raise ValueError
            return self.data
        except Exception:
            raise ValueError

    def vector_X(self):
        matrix1 = np.delete(self.Stock_prices, self.days-1, axis=0)
        matrix2 = np.delete(self.Stock_prices, 0, axis=0)
        return np.true_divide(matrix2, matrix1)

    def v_x(self):
        self.x = self.data.to_numpy()
        self.x = self.x[1:]/self.x[:-1]
        return self.x

    def S_function(self, b, x):
        S = np.ones(self.days, dtype=float)
        for i in range(len(S)-1):
            S[i+1] = S[i]*np.dot(b[i, :], x[i, :])
        return S

    def denominator(self, n, x, b, Scalar):
        sum = 0
        for k in range(self.company):
            sum = sum + b[k]*math.exp((n*x[k])/Scalar)
        return sum

    def Combinations(self, a, x):
        divi = range(a+1)
        w = np.asarray(list(filter(lambda x: sum(x) == a, self.comb(divi, len(x[0])))))
        return w*(1/a)

    def comb(self, z, k):
        if k == 1:
            return [[atom] for atom in z]
        combinations = []
        smaller = self.comb(z, k - 1)
        for num in z:
            for k in smaller:
                combinations.append([num] + k)
        return combinations

    def S_function2(self, x, b, t):
        self.s = np.prod(list(map(lambda i, j: np.dot(i, j), x, b))[:t+1])
        return float(self.s)


    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        """
        calculates the universal portfolio for the previously requested stocks

        :param int portfolio_quantization: size of discrete steps of between computed portfolios. each step has size 1/portfolio_quantization
        :return: returns a list of floats, representing the growth trading  per day"""
        a = portfolio_quantization
        x = self.v_x()
        b = [[1/len(x[0]) for i in x[0]]]
        Combi = np.asarray(self.Combinations(a, x))
        for t in range(len(x)):
            f = []; g = []
            for i in range(len(Combi)):
                table = np.repeat([Combi[i]], len(x[:t+1]), axis=0)
                f.append(Combi[i]*self.S_function2(x[:t+1], table, t))
                g.append(self.S_function2(x[:t+1], table, t))
            b.append(sum(f)/sum(g))
        S1 = [self.S_function2(x, b[:-1], t) for t in range(len(x))]
        return [1.] + S1

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:
        """
        calculates the exponential gradient portfolio for the previously requested stocks
        :param float learn_rate: the learning rate of the algorithm, defaults to 0.5
        :return: is a list of floats, representing the growth trading  per day
        """
        b_vector = np.true_divide(np.ones((self.days-1, self.company)), self.company)
        matrix_prices = self.vector_X()
        for t in range(self.days - 2):
            Scalar_multi = np.dot(matrix_prices[t, :], b_vector[t, :])
            is_bool = True
            for j in range(self.company):
                m = b_vector[t, j]*(math.exp((learn_rate*matrix_prices[t, j])/Scalar_multi))
                if is_bool:
                    n = self.denominator(learn_rate, matrix_prices[t, j:], b_vector[t, j:], Scalar_multi)
                    is_bool = False
                b_vector[t+1, j] = m/n
        return self.S_function(b_vector, matrix_prices)

if __name__ == '__main__':                                             # You should keep this line for our auto-grading code.
    # t0 = time.time()
    main1 = PortfolioBuilder()
    shahaf = main1.get_daily_data(['GOOG', 'AAPL', 'MSFT'], date(2020, 1, 1), date(2020, 2, 1))
    universal1 = main1.find_universal_portfolio(20)
    print(universal1[:9])
    # print(universal1[10:18])
    # print(universal1[19:])
    # t1 = time.time()
    # print(t1 - t0)