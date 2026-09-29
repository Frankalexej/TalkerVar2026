import unittest
import numpy as np
from sklearn.metrics import silhouette_samples
from lab_projects.vowel_geometry.metrics import geometry,query_silhouettes


class MetricTests(unittest.TestCase):
    def test_four_feature_config_only_changes_features_and_output(self):
        from lab_projects.vowel_geometry.config import CONFIG as old
        from lab_projects.vowel_geometry.config_four import CONFIG as new
        self.assertEqual(new['features'], ['f1_hz','f2_hz','f3_hz','duration_s'])
        for key in old:
            if key not in ('features','output_dir'):
                self.assertEqual(old[key],new[key])

    def test_four_dimensional_queries_match_sklearn(self):
        rng = np.random.default_rng(2026)
        x = rng.normal(size=(120,4)); y = np.arange(120)%5
        queries = rng.choice(120,25,replace=False)
        np.testing.assert_allclose(query_silhouettes(x,{'truth':y},queries,7)['truth'],
                                   silhouette_samples(x,y)[queries],atol=1e-12)

    def test_exact_queries_match_sklearn(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(75,5)); y = np.repeat(np.arange(5),15)
        q = rng.choice(75,23,replace=False)
        p = rng.permutation(y)
        out = query_silhouettes(x,{'a':y,'b':p},q,7)
        for key,labels in [('a',y),('b',p)]:
            np.testing.assert_allclose(out[key],silhouette_samples(x,labels)[q],atol=1e-12)

    def test_duplicates_and_singletons(self):
        x = np.array([[0,0],[0,0],[1,1],[1,2],[10,10]],dtype=float)
        y = np.array([0,0,1,1,2]); q = np.arange(5)
        np.testing.assert_allclose(query_silhouettes(x,{'a':y},q,2)['a'],silhouette_samples(x,y),atol=1e-12)

    def test_variance_ratio_matches_ch(self):
        x = np.random.default_rng(4).normal(size=(100,5)); y = np.arange(100)%4
        r = geometry(x,y); f = r['between_fraction']
        self.assertAlmostEqual(r['calinski_harabasz'],(f/(1-f))*96/3,places=10)

    def test_macro_and_weighted_aggregation_are_distinct(self):
        import pandas as pd
        from lab_projects.vowel_geometry.report import aggregate,MEASURES,VIEWS
        rows=[]
        for view in VIEWS:
            for group,n,value in [('small',10,.1),('large',90,.9)]:
                for seed in [42,43,44]:
                    rows.append({'view':view,'group':group,'seed':seed,'n':n,'k':2,'n_fit':n,'n_query':n,
                                 **{metric:value for metric in MEASURES}})
        _,summary=aggregate(pd.DataFrame(rows))
        macro=summary[summary.weighting.eq('macro')]
        weighted=summary[summary.weighting.eq('token_weighted')]
        np.testing.assert_allclose(macro['mean'],.5)
        np.testing.assert_allclose(weighted['mean'],.82)
        np.testing.assert_allclose(summary.repeat_sd,0,atol=1e-15)


if __name__=='__main__':
    unittest.main()
