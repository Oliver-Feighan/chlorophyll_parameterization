%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_18 TDDFT with PBE1PBE functional

0 1
Mg   34.219   49.175   25.277
C   34.300   47.512   28.270
C   32.536   51.861   26.968
C   33.270   50.508   22.306
C   35.332   46.497   23.675
N   33.444   49.574   27.382
C   33.805   48.805   28.427
C   33.418   49.442   29.791
C   32.710   50.782   29.349
C   32.893   50.775   27.827
C   31.218   51.001   29.782
C   34.640   49.619   30.796
C   34.439   49.296   32.253
C   33.036   49.335   32.777
O   32.171   48.481   32.651
O   32.784   50.444   33.592
N   33.148   50.992   24.751
C   32.611   52.021   25.503
C   31.999   52.981   24.637
C   32.127   52.589   23.316
C   32.872   51.262   23.396
C   31.433   54.222   25.171
C   31.698   53.388   22.115
O   32.101   53.085   20.940
C   30.961   54.709   22.165
N   34.182   48.568   23.257
C   33.798   49.268   22.204
C   34.251   48.530   20.765
C   34.908   47.289   21.333
C   34.790   47.452   22.865
C   33.057   48.194   19.847
C   36.372   46.967   20.760
C   37.633   47.662   21.437
N   34.713   47.263   25.818
C   35.223   46.290   25.076
C   35.621   45.145   25.907
C   35.217   45.530   27.180
C   34.703   46.868   27.070
C   36.236   43.869   25.467
C   35.230   45.257   28.614
O   35.675   44.243   29.176
C   34.606   46.464   29.378
C   33.414   45.970   30.129
O   32.439   45.472   29.559
O   33.773   45.911   31.453
C   32.909   45.083   32.340
C   31.457   50.546   34.137
C   31.167   52.026   34.109
C   30.305   52.710   34.890
C   29.425   52.052   35.904
C   30.064   54.142   34.587
C   30.908   55.157   35.400
C   31.419   56.352   34.575
C   31.234   57.681   35.455
C   32.487   58.626   35.100
C   29.956   58.519   35.183
C   29.015   58.604   36.423
C   29.219   59.878   37.368
C   28.207   61.017   37.019
C   28.982   62.379   37.107
C   26.881   60.912   37.899
C   25.863   62.050   37.583
C   25.300   62.839   38.822
C   25.252   64.358   38.465
C   25.197   65.217   39.708
C   24.091   64.719   37.480
H   32.045   52.632   27.566
H   33.111   51.051   21.372
H   35.781   45.677   23.112
H   32.718   48.786   30.308
H   33.270   51.633   29.739
H   30.904   52.012   30.042
H   31.068   50.362   30.651
H   30.518   50.757   28.983
H   35.105   50.605   30.782
H   35.429   48.998   30.370
H   35.117   49.944   32.808
H   34.858   48.292   32.313
H   31.837   55.076   24.626
H   31.575   54.320   26.247
H   30.345   54.233   25.097
H   31.599   55.455   22.639
H   30.091   54.548   22.803
H   30.752   55.029   21.144
H   34.928   49.300   20.396
H   34.366   46.383   21.064
H   33.004   47.123   19.654
H   33.219   48.772   18.936
H   32.214   48.522   20.454
H   36.262   47.312   19.732
H   36.442   45.879   20.751
H   38.390   47.939   20.703
H   38.018   46.917   22.134
H   37.325   48.592   21.914
H   35.511   43.233   24.958
H   36.550   43.364   26.380
H   37.105   44.076   24.842
H   35.318   46.794   30.134
H   31.859   45.136   32.053
H   32.987   45.562   33.316
H   33.256   44.051   32.294
H   31.424   50.098   35.130
H   30.743   49.983   33.536
H   31.718   52.699   33.452
H   28.374   52.029   35.617
H   29.474   52.729   36.757
H   29.711   51.058   36.246
H   29.023   54.195   34.907
H   30.193   54.311   33.518
H   31.650   54.541   35.909
H   30.276   55.564   36.189
H   30.884   56.426   33.628
H   32.476   56.169   34.383
H   31.468   57.454   36.495
H   33.489   58.195   35.080
H   32.492   59.506   35.743
H   32.388   58.952   34.064
H   29.435   58.122   34.312
H   30.236   59.546   34.952
H   29.279   57.757   37.056
H   27.978   58.464   36.117
H   30.222   60.193   37.080
H   29.159   59.449   38.369
H   27.872   60.905   35.988
H   28.927   62.854   36.127
H   30.025   62.177   37.352
H   28.577   63.150   37.762
H   27.148   60.807   38.950
H   26.424   59.950   37.664
H   24.941   61.663   37.150
H   26.290   62.746   36.862
H   26.039   62.673   39.606
H   24.366   62.356   39.111
H   26.167   64.554   37.906
H   25.952   66.002   39.660
H   25.236   64.665   40.647
H   24.225   65.708   39.756
H   24.486   65.236   36.605
H   23.284   65.344   37.861
H   23.610   63.777   37.216

