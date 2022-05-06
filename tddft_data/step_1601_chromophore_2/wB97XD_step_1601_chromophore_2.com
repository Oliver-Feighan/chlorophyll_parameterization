%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1601_chromophore_2 TDDFT with wB97XD functional

0 1
Mg   2.494   -0.361   43.438
C   5.937   -0.000   43.454
C   2.146   2.629   41.784
C   -0.868   -1.214   42.912
C   3.029   -3.645   44.608
N   3.896   1.151   42.755
C   5.270   1.038   42.764
C   5.920   2.276   42.072
C   4.600   3.181   41.819
C   3.458   2.270   42.090
C   4.511   4.510   42.598
C   6.733   1.992   40.756
C   6.045   1.085   39.687
C   5.742   1.624   38.285
O   5.414   2.787   38.075
O   5.801   0.665   37.284
N   0.843   0.681   42.577
C   0.930   1.908   42.017
C   -0.412   2.275   41.675
C   -1.312   1.147   41.939
C   -0.465   0.079   42.440
C   -0.771   3.651   41.232
C   -2.803   1.184   41.532
O   -3.393   2.169   41.049
C   -3.628   -0.076   41.715
N   1.275   -2.141   43.558
C   -0.080   -2.247   43.398
C   -0.585   -3.587   43.838
C   0.714   -4.458   43.891
C   1.722   -3.341   44.017
C   -1.362   -3.554   45.199
C   1.012   -5.319   42.673
C   1.542   -6.729   43.011
N   4.166   -1.586   44.063
C   4.170   -2.848   44.593
C   5.487   -3.156   45.093
C   6.258   -2.086   44.601
C   5.375   -1.160   44.044
C   6.031   -4.389   45.852
C   7.639   -1.551   44.407
O   8.759   -2.032   44.745
C   7.500   -0.231   43.688
C   8.249   0.867   44.335
O   9.146   1.567   43.859
O   7.791   1.031   45.620
C   8.548   2.095   46.347
C   5.367   1.154   35.953
C   5.933   0.128   34.860
C   5.239   -0.863   34.197
C   3.790   -1.232   34.634
C   5.826   -1.722   33.091
C   5.218   -1.601   31.766
C   6.108   -0.602   30.985
C   5.217   0.319   29.969
C   6.209   0.732   28.871
C   4.492   1.526   30.698
C   2.960   1.465   30.638
C   2.161   1.329   31.992
C   0.961   2.245   32.107
C   0.591   2.378   33.588
C   -0.251   1.695   31.269
C   -0.514   2.626   30.000
C   -1.973   3.072   29.885
C   -2.085   4.409   29.072
C   -2.965   5.474   29.840
C   -2.565   4.212   27.657
H   2.002   3.556   41.225
H   -1.936   -1.441   42.926
H   3.134   -4.686   44.921
H   6.549   2.747   42.827
H   4.653   3.467   40.768
H   3.458   4.734   42.769
H   4.968   5.337   42.053
H   4.969   4.350   43.574
H   7.589   1.439   41.144
H   7.095   2.940   40.357
H   5.113   0.631   40.025
H   6.851   0.357   39.593
H   -0.773   3.804   40.152
H   -0.049   4.278   41.755
H   -1.747   3.974   41.594
H   -3.684   -0.340   42.772
H   -3.368   -0.922   41.078
H   -4.587   0.148   41.250
H   -1.247   -3.972   43.062
H   0.827   -5.091   44.771
H   -0.966   -4.139   46.029
H   -2.317   -3.980   44.894
H   -1.469   -2.563   45.642
H   1.560   -4.780   41.900
H   0.103   -5.586   42.133
H   2.374   -6.830   42.315
H   0.931   -7.621   42.871
H   1.972   -6.684   44.011
H   5.715   -4.234   46.884
H   7.112   -4.488   45.750
H   5.540   -5.230   45.362
H   7.852   -0.299   42.658
H   9.555   1.737   46.562
H   8.132   2.146   47.354
H   8.370   3.018   45.795
H   5.780   2.125   35.678
H   4.281   1.230   35.898
H   6.969   0.281   34.557
H   3.195   -0.675   33.910
H   3.520   -0.919   35.643
H   3.579   -2.299   34.553
H   5.707   -2.753   33.423
H   6.901   -1.551   33.140
H   4.229   -1.146   31.801
H   5.201   -2.651   31.474
H   6.813   -1.141   30.352
H   6.743   0.012   31.624
H   4.510   -0.407   29.567
H   6.457   -0.066   28.172
H   7.130   1.045   29.362
H   5.810   1.608   28.359
H   4.797   2.457   30.223
H   4.839   1.589   31.730
H   2.614   0.561   30.135
H   2.561   2.378   30.196
H   2.861   1.539   32.801
H   1.860   0.286   32.095
H   1.117   3.276   31.790
H   0.982   3.314   33.986
H   1.040   1.594   34.199
H   -0.470   2.331   33.833
H   -1.207   1.576   31.778
H   0.022   0.720   30.864
H   -0.356   1.928   29.178
H   0.227   3.409   29.834
H   -2.281   3.209   30.921
H   -2.520   2.206   29.512
H   -1.118   4.911   29.054
H   -3.964   5.423   29.407
H   -2.616   6.500   29.718
H   -3.034   5.370   30.923
H   -2.236   3.301   27.156
H   -2.127   5.041   27.102
H   -3.636   4.409   27.613

