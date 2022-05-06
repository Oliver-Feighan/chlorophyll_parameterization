%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_701_chromophore_22 TDDFT with blyp functional

0 1
Mg   8.478   48.047   25.309
C   6.479   48.244   28.014
C   11.206   48.839   27.376
C   10.582   47.849   22.584
C   5.818   47.702   23.193
N   8.838   48.364   27.524
C   7.812   48.351   28.433
C   8.380   48.726   29.817
C   9.719   49.419   29.429
C   10.001   48.859   28.063
C   9.685   50.983   29.387
C   8.617   47.405   30.660
C   8.510   47.545   32.153
C   7.807   46.506   33.006
O   7.122   45.568   32.656
O   8.130   46.833   34.273
N   10.507   48.322   25.015
C   11.487   48.500   25.993
C   12.798   48.462   25.423
C   12.670   48.044   24.113
C   11.231   48.060   23.854
C   14.047   48.975   26.139
C   13.715   47.729   23.071
O   13.506   47.437   21.908
C   15.169   47.778   23.395
N   8.234   47.942   23.092
C   9.219   47.765   22.219
C   8.725   47.466   20.829
C   7.211   47.258   21.026
C   7.082   47.568   22.527
C   9.028   48.602   19.837
C   6.598   45.843   20.693
C   5.416   45.762   19.699
N   6.597   47.930   25.524
C   5.556   47.905   24.591
C   4.293   47.928   25.196
C   4.650   47.945   26.601
C   6.041   48.078   26.691
C   2.950   47.891   24.617
C   4.061   48.036   27.889
O   2.822   48.039   28.107
C   5.196   48.243   28.922
C   5.053   47.102   29.872
O   5.656   46.080   29.764
O   4.226   47.400   30.940
C   4.002   46.388   32.023
C   7.692   45.823   35.219
C   8.204   46.320   36.562
C   7.623   46.300   37.816
C   6.275   45.688   38.054
C   8.282   46.862   39.169
C   9.080   45.816   39.983
C   10.617   46.099   40.235
C   10.831   46.667   41.668
C   10.721   48.211   41.765
C   12.215   46.102   42.194
C   12.028   44.709   42.741
C   12.447   44.574   44.245
C   11.345   44.326   45.302
C   11.634   43.076   46.119
C   11.018   45.498   46.159
C   10.281   46.587   45.385
C   9.666   47.755   46.244
C   10.449   49.214   46.176
C   11.524   49.345   47.225
C   9.538   50.433   46.130
H   11.980   49.113   28.095
H   11.276   47.723   21.750
H   4.916   47.579   22.590
H   7.624   49.394   30.229
H   10.569   49.031   29.991
H   8.706   51.413   29.602
H   9.876   51.217   28.340
H   10.525   51.459   29.893
H   9.606   47.016   30.417
H   7.838   46.714   30.339
H   8.136   48.513   32.486
H   9.528   47.504   32.540
H   13.961   49.130   27.214
H   14.448   49.933   25.808
H   14.832   48.225   26.048
H   15.521   46.907   23.947
H   15.388   48.707   23.922
H   15.681   47.818   22.434
H   9.227   46.513   20.660
H   6.547   47.961   20.523
H   8.838   49.592   20.250
H   8.347   48.542   18.988
H   10.047   48.519   19.456
H   6.350   45.257   21.578
H   7.408   45.328   20.177
H   5.501   44.827   19.146
H   5.377   46.607   19.012
H   4.498   45.879   20.275
H   2.732   48.951   24.485
H   2.220   47.387   25.251
H   3.013   47.354   23.671
H   4.941   49.202   29.375
H   2.948   46.246   32.262
H   4.571   46.754   32.879
H   4.354   45.451   31.591
H   8.269   44.936   34.958
H   6.606   45.756   35.159
H   9.202   46.757   36.524
H   6.312   44.924   38.829
H   5.900   45.123   37.200
H   5.499   46.390   38.360
H   7.509   47.361   39.753
H   8.984   47.652   38.900
H   9.099   44.907   39.382
H   8.554   45.628   40.919
H   10.891   46.734   39.393
H   11.120   45.160   40.002
H   10.071   46.425   42.411
H   10.042   48.582   42.533
H   10.373   48.574   40.798
H   11.638   48.711   42.078
H   12.553   46.792   42.967
H   13.062   45.949   41.526
H   12.682   43.999   42.235
H   11.009   44.351   42.593
H   12.993   45.442   44.616
H   13.229   43.816   44.195
H   10.426   44.132   44.747
H   12.603   42.653   45.854
H   10.909   42.309   45.846
H   11.527   43.159   47.200
H   11.885   45.969   46.624
H   10.239   45.262   46.884
H   9.474   46.258   44.730
H   11.063   47.075   44.804
H   9.771   47.523   47.304
H   8.590   47.873   46.119
H   11.131   49.202   45.326
H   12.394   48.729   46.998
H   11.122   49.100   48.208
H   11.863   50.352   47.470
H   9.479   51.065   47.016
H   8.480   50.211   45.994
H   9.942   51.120   45.387

