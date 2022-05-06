%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1701_chromophore_20 TDDFT with PBE1PBE functional

0 1
Mg   6.641   57.172   41.878
C   5.756   53.657   41.658
C   9.536   56.500   40.484
C   7.054   60.511   41.650
C   3.600   57.686   43.558
N   7.429   55.229   41.028
C   6.921   53.990   40.969
C   8.066   53.056   40.469
C   9.354   54.022   40.401
C   8.751   55.363   40.741
C   10.546   53.665   41.276
C   7.832   52.297   39.023
C   8.540   50.936   38.937
C   8.036   49.845   37.929
O   7.708   48.695   38.206
O   7.688   50.384   36.701
N   8.096   58.403   41.025
C   9.228   57.912   40.542
C   10.086   59.023   40.197
C   9.355   60.255   40.455
C   8.105   59.744   41.078
C   11.510   58.984   39.510
C   9.766   61.673   39.982
O   10.879   61.807   39.501
C   8.818   62.844   39.970
N   5.473   58.905   42.277
C   5.784   60.195   42.185
C   4.777   61.150   42.774
C   3.658   60.224   43.272
C   4.232   58.830   43.047
C   5.390   62.085   43.868
C   2.233   60.440   42.712
C   1.169   60.726   43.786
N   5.006   55.964   42.557
C   3.866   56.269   43.306
C   3.125   55.159   43.703
C   3.790   54.093   43.040
C   4.935   54.611   42.379
C   1.918   55.100   44.600
C   3.813   52.691   42.744
O   2.947   51.839   42.961
C   5.175   52.354   42.065
C   4.931   51.379   40.981
O   4.363   51.614   39.946
O   5.375   50.207   41.445
C   5.212   49.107   40.524
C   7.650   49.360   35.628
C   7.186   50.093   34.371
C   5.932   50.401   34.041
C   4.769   50.031   34.867
C   5.685   51.028   32.711
C   6.529   52.276   32.394
C   6.015   53.126   31.165
C   5.739   54.627   31.627
C   4.298   54.820   31.528
C   6.641   55.628   30.870
C   8.105   55.416   31.293
C   8.991   55.036   30.064
C   10.412   55.778   30.131
C   10.840   56.136   28.692
C   11.417   54.849   30.716
C   12.689   55.655   31.368
C   13.934   55.325   30.583
C   14.669   56.598   29.978
C   14.418   56.651   28.457
C   16.240   56.663   30.233
H   10.540   56.211   40.167
H   7.169   61.593   41.563
H   2.757   57.914   44.214
H   8.219   52.311   41.250
H   9.703   54.118   39.373
H   10.445   52.741   41.846
H   10.678   54.501   41.964
H   11.397   53.435   40.635
H   8.195   52.935   38.217
H   6.831   52.212   38.600
H   8.602   50.652   39.987
H   9.596   51.035   38.684
H   11.424   59.417   38.514
H   11.882   57.962   39.432
H   12.226   59.548   40.107
H   7.872   62.658   39.461
H   9.209   63.649   39.347
H   8.505   63.211   40.948
H   4.353   61.779   41.991
H   3.545   60.350   44.349
H   4.878   61.895   44.811
H   5.162   63.076   43.475
H   6.471   62.014   43.989
H   1.995   59.490   42.234
H   2.245   61.256   41.989
H   0.226   60.420   43.334
H   0.994   61.788   43.956
H   1.381   60.191   44.712
H   2.148   54.449   45.443
H   1.134   54.534   44.096
H   1.595   56.096   44.903
H   5.764   51.981   42.904
H   4.220   48.721   40.754
H   5.978   48.370   40.763
H   5.198   49.276   39.447
H   6.929   48.545   35.695
H   8.658   48.954   35.540
H   8.016   50.225   33.676
H   4.984   49.444   35.759
H   4.132   49.429   34.218
H   4.329   50.957   35.239
H   4.635   51.205   32.477
H   6.117   50.258   32.071
H   7.559   52.047   32.120
H   6.513   52.879   33.302
H   5.317   52.768   30.408
H   6.894   53.175   30.522
H   5.898   54.908   32.668
H   3.922   54.583   30.532
H   4.094   55.865   31.762
H   3.704   54.260   32.250
H   6.497   56.658   31.196
H   6.443   55.554   29.800
H   8.271   54.663   32.063
H   8.555   56.302   31.741
H   8.528   55.284   29.109
H   9.216   53.970   30.085
H   10.440   56.682   30.739
H   11.348   55.359   28.121
H   11.525   56.981   28.758
H   10.061   56.358   27.963
H   11.772   54.209   29.909
H   10.834   54.331   31.477
H   12.716   55.501   32.446
H   12.512   56.729   31.299
H   13.909   54.563   29.804
H   14.585   54.826   31.301
H   14.142   57.443   30.419
H   15.172   57.129   27.831
H   13.542   57.218   28.144
H   14.322   55.624   28.104
H   16.749   57.084   29.366
H   16.597   55.667   30.495
H   16.468   57.219   31.142

