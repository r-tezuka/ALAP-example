import numpy as np
# Ref: https://algorithm.joho.info/programming/python-numpy-matrix-cholesky/


def main():

    A = np.array([[2., 1., 3]    # 行列Aの生成
                 , [2., 2, 6.]])

    L = np.linalg.cholesky(np.dot(A, A.T))      # 行列AをQR分解
    # 結果を表示
    print("A=\n", A)
    print("L=\n", L)
    print("L*L^T=\n", L.dot(L.T))


if __name__ == '__main__':
    main()
