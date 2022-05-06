%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_701_chromophore_14 TDDFT with cam-b3lyp functional

0 1
Mg   46.852   44.643   44.017
C   43.394   44.220   43.526
C   47.225   41.203   43.159
C   50.139   45.009   43.530
C   46.329   48.106   43.994
N   45.500   42.985   43.330
C   44.110   43.068   43.279
C   43.515   41.713   43.023
C   44.694   40.743   42.974
C   45.892   41.676   43.198
C   44.610   39.542   43.934
C   42.543   41.538   41.803
C   43.132   42.233   40.573
C   42.903   41.547   39.210
O   42.186   40.601   38.949
O   43.695   42.202   38.318
N   48.511   43.293   43.707
C   48.448   41.912   43.423
C   49.809   41.382   43.228
C   50.709   42.517   43.435
C   49.794   43.670   43.671
C   50.173   40.027   42.887
C   52.202   42.443   43.471
O   52.830   41.389   43.330
C   53.096   43.584   43.851
N   48.040   46.379   43.692
C   49.392   46.233   43.496
C   50.092   47.577   43.170
C   48.830   48.534   43.398
C   47.655   47.637   43.668
C   51.354   47.681   44.178
C   48.550   49.643   42.309
C   48.121   51.130   42.712
N   45.189   45.922   43.861
C   45.120   47.268   44.023
C   43.795   47.716   44.222
C   42.977   46.505   43.998
C   43.958   45.485   43.696
C   43.395   49.140   44.493
C   41.651   45.889   43.912
O   40.547   46.387   43.944
C   41.890   44.431   43.594
C   41.248   43.598   44.656
O   40.311   42.781   44.508
O   41.807   43.913   45.895
C   41.171   43.222   47.060
C   43.725   41.607   36.986
C   44.866   42.246   36.263
C   45.794   41.735   35.496
C   45.800   40.267   35.014
C   46.812   42.569   34.814
C   46.338   43.524   33.625
C   47.084   43.354   32.325
C   46.123   43.725   31.080
C   45.145   44.916   31.238
C   45.329   42.400   30.674
C   45.706   41.937   29.219
C   44.646   42.498   28.232
C   45.105   43.094   26.855
C   44.276   44.301   26.588
C   45.086   42.040   25.666
C   46.424   41.982   24.843
C   46.235   42.012   23.325
C   47.089   43.288   22.735
C   46.548   43.674   21.383
C   48.605   43.017   22.943
H   47.276   40.181   42.778
H   51.146   45.196   43.153
H   46.240   49.182   44.158
H   42.910   41.456   43.892
H   44.902   40.294   42.002
H   44.533   38.707   43.237
H   43.685   39.545   44.510
H   45.466   39.460   44.604
H   41.601   41.971   42.138
H   42.520   40.462   41.628
H   44.211   42.334   40.689
H   42.649   43.186   40.791
H   50.187   39.334   43.728
H   51.188   39.878   42.519
H   49.409   39.664   42.199
H   52.724   44.210   44.663
H   53.089   44.337   43.063
H   54.079   43.254   44.186
H   50.483   47.575   42.153
H   49.185   49.118   44.247
H   52.305   47.732   43.647
H   51.381   46.867   44.901
H   51.338   48.503   44.894
H   47.623   49.354   41.816
H   49.332   49.550   41.555
H   48.509   51.743   41.899
H   48.655   51.450   43.607
H   47.054   51.227   42.913
H   42.313   49.041   44.574
H   43.575   49.827   43.666
H   43.918   49.372   45.421
H   41.335   44.326   42.662
H   40.099   43.361   47.202
H   41.691   43.415   47.999
H   41.199   42.158   46.823
H   42.801   41.793   36.438
H   43.844   40.523   37.014
H   45.052   43.252   36.638
H   44.903   39.762   35.374
H   46.731   39.896   35.442
H   45.746   40.142   33.933
H   47.598   41.904   34.457
H   47.202   43.245   35.575
H   46.466   44.533   34.017
H   45.286   43.244   33.570
H   47.315   42.292   32.415
H   47.949   44.011   32.236
H   46.810   44.063   30.303
H   44.090   44.644   31.261
H   45.433   45.544   30.395
H   45.269   45.488   32.158
H   44.270   42.658   30.714
H   45.422   41.609   31.418
H   45.617   40.850   29.196
H   46.675   42.222   28.809
H   44.063   43.228   28.794
H   44.003   41.658   27.968
H   46.135   43.405   27.031
H   44.535   45.156   27.212
H   43.241   44.094   26.857
H   44.305   44.683   25.567
H   44.165   42.151   25.093
H   45.049   41.051   26.122
H   46.988   41.078   25.075
H   47.054   42.834   25.098
H   45.159   41.997   23.150
H   46.587   41.134   22.783
H   46.863   44.042   23.489
H   45.971   42.837   20.990
H   47.379   43.953   20.734
H   45.873   44.515   21.539
H   48.957   43.823   23.588
H   49.075   43.100   21.963
H   48.704   41.984   23.277

