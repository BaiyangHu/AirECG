import numpy as np

def second_dev(data, h):
    second_derivative = np.zeros_like(data)
    N = len(data)

    for i in range(3, N-3):
        s0 = data[i]
        s_minus_3 = data[i-3]
        s_minus_2 = data[i-2]
        s_minus_1 = data[i-1]
        s_plus_1 = data[i+1]
        s_plus_2 = data[i+2]
        s_plus_3 = data[i+3]

        # Using the equation given in the paper
        second_derivative[i] = ((s_minus_3 + s_plus_3) +
                                2 * (s_minus_2 + s_plus_2) -
                                (s_minus_1 + s_plus_1) -
                                4 * s0) / (16 * h**2)

    return second_derivative

if __name__ == '__main__':
    data = np.random.rand(10)
    h = 0.01
    second_derivative = second_dev(data, h)
    print(second_derivative)