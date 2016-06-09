import unittest

import numpy as np


class test_CellStar(unittest.TestCase):
    def test_fill_holes(self):
        import contrib.cell_star.utils.image_util as image_util

        a = np.ones((30, 30), dtype=int)
        a[10:20, 20:25] = 0
        expected = a.copy()
        a[0:3, 0:3] = 0

        res = image_util.fill_holes(a, 3, 15)

        self.assertTrue((expected == res).all())

    def test_random_seeds(self):
        from contrib.cell_star.core.seed import Seed
        from contrib.cell_star.core.point import Point
        from contrib.cell_star.core.seeder import Seeder
        seed1 = Seed(10, 10, 'test')
        seed2 = Seed(100, 100, 'test2')
        new_seeds = Seeder.rand_seeds(5, 2, [seed1, seed2])
        self.assertEqual(4, len(new_seeds))

        seed3 = Seed(0, 0, 'test3')
        new_zero_seeds = Seeder.rand_seeds(10, 10, [seed3])
        self.assertTrue(all([seed.euclidean_distance_to(Point(0, 0)) < 10 for seed in new_zero_seeds]))

    def test_random_seeds_fractions(self):
        from contrib.cell_star.core.seed import Seed
        from contrib.cell_star.core.seeder import Seeder
        np.random.seed(1)
        def assertIsClose(x, y, seed):
            self.assertEqual(x, int(seed.x + .5))
            self.assertEqual(y, int(seed.y + .5))

        def run_and_sort(times, seeds):
            return sorted(Seeder.rand_seeds(0.1, times, seeds), key=lambda x: x.x)


        seeds = []
        seeds.append(Seed(1, 1, 'test1'))
        seeds.append(Seed(2, 2, 'test2'))
        seeds.append(Seed(3, 3, 'test3'))
        seeds.append(Seed(4, 4, 'test4'))

        new_seeds = run_and_sort(1.6, seeds)

        self.assertEqual(6, len(new_seeds))
        # on seed 1
        assertIsClose(1, 1, new_seeds[0])
        assertIsClose(2, 2, new_seeds[1])
        assertIsClose(3, 3, new_seeds[2])
        assertIsClose(3, 3, new_seeds[3])
        assertIsClose(4, 4, new_seeds[4])
        assertIsClose(4, 4, new_seeds[5])

        new_seeds = run_and_sort(0.3, seeds)
        self.assertEqual(1, len(new_seeds))
        assertIsClose(4, 4, new_seeds[0])

        new_seeds = run_and_sort(2.5, seeds)
        self.assertEqual(10, len(new_seeds))
        assertIsClose(4, 4, new_seeds[7])
        assertIsClose(4, 4, new_seeds[9])

        new_seeds = run_and_sort(1.99, seeds)
        self.assertEqual(7, len(new_seeds))

        new_seeds = run_and_sort(0.01, seeds)
        self.assertEqual(0, len(new_seeds))

        new_seeds = run_and_sort(3, seeds)
        self.assertEqual(3, len([s for s in new_seeds if int(s.x + .5) == 1]))
        self.assertEqual(3, len([s for s in new_seeds if int(s.x + .5) == 2]))
        self.assertEqual(3, len([s for s in new_seeds if int(s.x + .5) == 3]))
        self.assertEqual(3, len([s for s in new_seeds if int(s.x + .5) == 4]))

    def test_loop_connected(self):
        from contrib.cell_star.utils import calc_util

        def validate(exp, res):
            self.assertTrue((exp[0] == res[0]).all())
            self.assertTrue((exp[1] == res[1]).all())
            self.assertTrue((exp[2] == res[2]).all())

        case_1 = np.array([0, 0, 0, 1, 0])
        exp_1 = np.array([1]), [3], [3]
        res_1 = calc_util.loop_connected_components(case_1)
        self.assertEqual(exp_1, res_1)

        case_2 = np.array([0, 0, 1, 1, 0])
        exp_2 = np.array([2]), [2], [3]
        res_2 = calc_util.loop_connected_components(case_2)
        self.assertEqual(exp_2, res_2)

        case_3 = np.array([1, 0, 0, 1, 1])
        exp_3 = np.array([3]), [3], [0]
        res_3 = calc_util.loop_connected_components(case_3)
        self.assertEqual(exp_3, res_3)

        case_4 = np.array([1, 1, 0, 1, 1, 0])
        exp_4 = np.array([2, 2]), np.array([0, 3]), np.array([1, 4])
        res_4 = calc_util.loop_connected_components(case_4)
        validate(exp_4, res_4)

        case_5 = np.array([1, 1, 0, 1, 0, 1])
        exp_5 = np.array([3, 1]), np.array([5, 3]), np.array([1, 3])
        res_5 = calc_util.loop_connected_components(case_5)
        validate(exp_5, res_5)

    def test_interpolate(self):
        from contrib.cell_star.utils import calc_util

        mask_1 = [False, False, True, False, True, True, False]
        values_1 = [2134,1564,10,3234,14,6,5345]
        res_1 = list(values_1)
        calc_util.interpolate_radiuses_old(mask_1, len(mask_1), res_1)
        self.assertEqual([8, 9, 10, 12, 14, 6, 7], res_1)

        res_1 = list(values_1)
        calc_util.interpolate_radiuses(mask_1, len(mask_1), res_1)
        self.assertEqual([8, 9, 10, 12, 14, 6, 7], res_1)