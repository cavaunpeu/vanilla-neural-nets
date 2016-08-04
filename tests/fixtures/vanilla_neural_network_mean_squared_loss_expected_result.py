import numpy as np


MEAN_SQUARED_LOSS_EXPECTED_RESULT = np.array([[  2.49498688e-02,   8.96294699e-03,   2.21902980e-02,
          3.12762312e-02,   3.57707165e-02,   7.32487370e-03,
          8.58245319e-01,   1.91780249e-02,   1.30322312e-02,
          4.09889796e-01],
       [  6.18845133e-02,   8.39734695e-04,   1.17994804e-02,
          1.59202607e-01,   1.58511897e-01,   1.00739567e-02,
          1.09242731e-02,   1.95280271e-03,   1.29301866e-02,
          2.32665356e-02],
       [  1.98716210e-02,   8.57596743e-03,   1.97030444e-04,
          1.42619477e-03,   4.63903765e-03,   2.50062731e-02,
          3.15715520e-02,   7.91870631e-02,   3.89438193e-03,
          1.86344002e-02],
       [  4.92838603e-04,   6.45264630e-02,   7.82399371e-04,
          6.95038023e-02,   6.06260662e-02,   1.82159474e-01,
          6.40507737e-02,   7.94461580e-03,   2.64676543e-03,
          6.03331178e-03],
       [  1.59010941e-02,   1.06939745e-02,   3.90722424e-02,
          2.89024906e-01,   1.32985171e-01,   1.11300761e-01,
          1.66433886e-01,   2.92014631e-02,   2.27515314e-03,
          3.42678217e-02],
       [  1.93598541e-02,   3.21067395e-02,   1.73583784e-03,
          3.55345974e-03,   3.87573410e-03,   1.22839740e-01,
          1.57904941e-02,   4.17981016e-02,   5.08267999e-03,
          3.96543949e-02],
       [  2.04277507e-02,   1.34964543e-02,   2.02189273e-02,
          4.89372706e-02,   6.47951573e-02,   2.03613524e-01,
          8.19435448e-03,   6.52455484e-04,   1.91536663e-02,
          6.42752001e-03],
       [  1.55759794e-02,   4.68455834e-01,   4.79540172e-03,
          3.36491366e-03,   2.19974618e-02,   3.63233301e-02,
          4.39720505e-01,   8.67091144e-02,   1.27343649e-01,
          3.01691285e-01],
       [  4.47045743e-03,   2.55046150e-04,   2.55911925e-02,
          1.26621007e-01,   8.97454898e-03,   4.64187222e-02,
          1.64904048e-01,   9.25577437e-03,   1.19141630e-03,
          1.31547563e-01],
       [  6.46863052e-02,   3.96664492e-03,   2.21241363e-04,
          9.12097560e-03,   3.32882345e-02,   2.06539436e-03,
          4.04142028e-01,   1.56047969e-02,   3.09235990e-02,
          1.98756142e-01],
       [  4.92179600e-01,   1.46407695e-01,   1.01754935e-04,
          3.63299906e-02,   4.07513545e-01,   1.22756214e-01,
          4.24848431e-02,   3.41891244e-02,   4.91168723e-02,
          1.67779059e-02],
       [  1.87424763e-02,   2.63704328e-03,   2.01536004e-04,
          4.18394365e-02,   5.83826021e-03,   2.87801691e-02,
          9.90327364e-02,   9.51880623e-03,   1.02686556e-01,
          8.46248622e-03],
       [  3.16382988e-03,   9.70264386e-03,   1.57145855e-03,
          9.02189453e-03,   2.58062872e-03,   1.27937534e-02,
          2.17591142e-01,   1.55289916e-03,   9.24359039e-02,
          1.49720392e-01],
       [  5.63222925e-02,   7.36708383e-03,   1.52954540e-04,
          3.16642872e-02,   1.22509878e-01,   3.54913530e-01,
          2.49958641e-02,   2.84067542e-03,   6.00816861e-02,
          1.33826314e-01],
       [  2.58769821e-02,   2.40555206e-01,   7.42606398e-04,
          1.52354013e-01,   4.93384415e-02,   1.14545648e-02,
          8.41337607e-02,   8.33391935e-03,   3.29147633e-03,
          1.50420085e-02],
       [  1.20345098e-01,   1.86985352e-02,   3.88074966e-03,
          1.00000663e-01,   2.98307643e-02,   8.74212335e-03,
          7.03443498e-03,   2.93054351e-02,   5.75424151e-03,
          8.37264162e-04],
       [  1.62666832e-02,   6.18909689e-02,   2.43400472e-01,
          1.81498225e-01,   5.29884878e-02,   6.52826981e-02,
          6.10067642e-02,   7.04380612e-02,   7.59389512e-04,
          8.30637975e-01],
       [  3.85464001e-02,   1.38377306e-02,   9.04079125e-02,
          6.76006870e-02,   1.90451607e-01,   1.53110972e-02,
          7.11777132e-01,   9.97133272e-02,   2.70062419e-03,
          3.56934687e-02],
       [  5.88118064e-02,   1.83492377e-03,   9.50161615e-03,
          1.91830318e-02,   1.65214947e-02,   2.05704069e-02,
          1.62836408e-01,   1.40525388e-04,   6.11417682e-03,
          1.11869167e-02],
       [  3.06162574e-03,   2.52466730e-02,   1.07013107e-02,
          1.25469275e-02,   1.26966633e-03,   1.62121273e-02,
          6.08842905e-03,   7.84851448e-03,   1.92067537e-02,
          5.44144787e-01],
       [  1.35579117e-03,   2.59730503e-03,   1.00122291e-03,
          1.03772881e-02,   1.01661166e-02,   2.04412385e-02,
          6.90367741e-02,   2.39440713e-03,   4.92711769e-03,
          1.84874196e-01],
       [  1.08526486e-03,   1.74318088e-01,   2.86182500e-04,
          1.51364905e-01,   1.57331859e-02,   1.88272308e-02,
          7.40511985e-02,   1.56779095e-03,   7.15316434e-02,
          9.84881395e-02],
       [  1.12200786e-03,   8.42534673e-03,   3.10489146e-03,
          3.16623903e-02,   2.58920102e-03,   1.39705108e-02,
          2.51314417e-01,   8.75639391e-03,   4.49954450e-02,
          4.24606500e-02],
       [  1.32945043e-03,   1.03317134e-02,   2.15854233e-02,
          9.60246623e-02,   1.53087304e-02,   1.11145834e-01,
          5.49105727e-03,   3.73377525e-03,   8.68456135e-03,
          1.31238717e-01],
       [  1.60348964e-02,   1.34660275e-01,   1.85551805e-03,
          7.75089023e-02,   6.04038151e-02,   4.38979663e-02,
          2.15559874e-01,   2.03307877e-01,   9.34691581e-02,
          5.77334700e-01]])
