import os

import pytest

from seyfert.fisher.linalg_tools import *


@pytest.fixture(scope='module')
def random_matrix():
    def _random_mat(m, n):
        return np.random.random((m, n))

    return _random_mat


@pytest.fixture(scope='module')
def random_symmetric_matrix(random_matrix):
    def _random_sym_mat(n):
        X = random_matrix(n, n)
        X = (X + X.T)/2

        return X

    return _random_sym_mat


@pytest.mark.skipif(os.getenv("USER", "") == "lucapaganin", reason="Too long to execute in local")
class TestLinalg:
    @pytest.mark.parametrize('symmetric', [True, False], ids=['symm', 'not_symm'])
    def test_enumeration(self, symmetric):
        if symmetric:
            shape = (10, 10)
            nrows, ncols = shape
            expected = {}
            cnt = 0
            for j in range(ncols):
                for i in range(j + 1):
                    expected[(i, j)] = cnt
                    cnt += 1
            result = get_enumeration_map(shape, symmetric)
        else:
            shape = (10, 4)
            nrows, ncols = shape
            expected = {}
            cnt = 0
            for j in range(ncols):
                for i in range(nrows):
                    expected[(i, j)] = cnt
                    cnt += 1
            result = get_enumeration_map(shape, symmetric)

        assert expected == result

    @pytest.mark.parametrize('n', np.random.randint(2, 200, 20))
    def test_vec_unvec(self, random_matrix, n):
        X = random_matrix(n, n)
        assert np.all(unvec(vec(X)) == X)

    @pytest.mark.parametrize('n', np.random.randint(2, 200, 20))
    def test_vecp_unvecp(self, random_symmetric_matrix, n):
        X = random_symmetric_matrix(n)
        assert np.all(unvecp(vecp(X)) == X)

    @pytest.mark.parametrize('n', np.random.randint(2, 60, 20))
    def test_duplication_matrix(self, random_symmetric_matrix, n):
        X = random_symmetric_matrix(n)
        vecX = vec(X)
        vecpX = vecp(X)

        D = duplication_matrix(n)

        assert np.all(vecX == D @ vecpX)

    @pytest.mark.parametrize('n', np.random.randint(2, 60, 20))
    def test_elimination_matrix(self, random_symmetric_matrix, n):
        X = random_symmetric_matrix(n)
        vecX = vec(X)
        vecpX = vecp(X)

        L = elimination_matrix(n)

        assert np.all(L @ vecX == vecpX)

    @pytest.mark.parametrize('n', np.random.randint(2, 60, 20))
    def test_dupl_mat_pseudoinverse(self, random_symmetric_matrix, n):
        X = random_symmetric_matrix(n)
        vecX = vec(X)
        vecpX = vecp(X)

        D = duplication_matrix(n)

        assert pytest.approx(np.linalg.pinv(D) @ vecX, abs=1e-10, rel=1e-10) == vecpX

    @pytest.mark.parametrize('n', np.random.randint(2, 60, 20))
    def test_transition_matrix(self, random_symmetric_matrix, n):
        X = random_symmetric_matrix(n)
        vecX = vec(X)
        vecpX = vecp(X)

        B = transition_matrix(n)

        assert pytest.approx(B.T @ vecX, abs=1e-10, rel=1e-10) == vecpX

    @pytest.mark.parametrize("m, n", [(m, n) for m, n in zip(np.random.randint(2, 100, 20),
                                                             np.random.randint(2, 100, 20))])
    def test_commut_matrix(self, random_matrix, m, n):
        K = commutation_matrix(m, n)
        X = random_matrix(m, n)

        assert np.all(vec(X.T) == K @ vec(X))

