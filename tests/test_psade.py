from .context import data, algorithms, pset, objective, config

import shutil
from copy import deepcopy
from numpy import any

class TestPSADE:
    def __init__(self):
        pass

    @classmethod
    def setup_class(cls):
        cls.data1s = [
            '# time    v1_result    v2_result    v3_result\n',
            ' 1 2.1   3.1   6.1\n',
        ]
        cls.d1s = data.Data()
        cls.d1s.data = cls.d1s._read_file_lines(cls.data1s, '\s+')

        # Note mutation_rate is set to 1.0 because for tests with few params, with a lower mutation_rate might randomly
        # create a duplicate parameter set, causing the "not in individuals" tests to fail.
        cls.config = config.Configuration({
            'population_size': 20, 'max_iterations': 20, 'fit_type': 'psade',
            ('uniform_var', 'v1__FREE'): [0, 10], ('uniform_var', 'v2__FREE'): [0, 10], ('uniform_var', 'v3__FREE'): [0, 10],
            'models': {'bngl_files/parabola.bngl'}, 'exp_data': {'bngl_files/par1.exp'}, 'initialization': 'lh',
            'bngl_files/parabola.bngl': ['bngl_files/par1.exp'],
            'output_dir': 'test_init'})

    @classmethod
    def teardown_class(cls):
        shutil.rmtree('test_init')

    def test_start(self):
        psade = algorithms.PSADE(self.config)
        assert psade.weight_max == 1.5
        start_params = psade.start_run()
        assert len(start_params) == 20
        assert len(psade.individuals) == 20

    def test_updates(self):
        psade = algorithms.PSADE(self.config)
        start_params = psade.start_run()
        # Run iteration 1
        for i in range(19):
            res = algorithms.Result(start_params[i], self.data1s, start_params[i].name)
            res.score = 42.+i*0.05 # we need different scores so that Tmax is not zero
            torun = psade.got_result(res)
            assert torun == []
        res = algorithms.Result(start_params[19], self.data1s, start_params[19].name)
        res.score = 43.
        torun = psade.got_result(res)
        assert torun == psade.individuals

        # Run iteration 2
        for i in range(20):
            res = algorithms.Result(psade.individuals[i], self.data1s, start_params[i].name)
            res.score = 42.+i*0.051
            torun = psade.got_result(res)
        assert len(psade.individuals) == 20
        assert len(psade.local_search_points) > 0 # The odds of this test failing are ~1E-60


