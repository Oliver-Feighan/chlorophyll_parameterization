%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1851_chromophore_11 TDDFT with blyp functional

0 1
Mg   53.878   23.513   44.321
C   51.162   25.963   44.100
C   51.323   21.060   44.165
C   56.229   21.371   44.165
C   56.120   26.147   44.188
N   51.590   23.548   44.061
C   50.733   24.622   44.011
C   49.320   24.137   43.871
C   49.342   22.700   44.366
C   50.839   22.367   44.182
C   48.812   22.501   45.768
C   48.685   24.369   42.484
C   49.500   23.906   41.204
C   48.867   22.673   40.490
O   48.681   21.515   40.969
O   48.463   23.004   39.208
N   53.820   21.540   44.176
C   52.674   20.720   44.230
C   53.188   19.324   44.322
C   54.596   19.346   44.209
C   54.933   20.794   44.198
C   52.318   18.142   44.469
C   55.485   18.129   44.204
O   54.963   17.009   44.093
C   56.959   18.178   44.181
N   55.887   23.737   44.051
C   56.729   22.681   44.170
C   58.211   23.072   44.358
C   58.134   24.580   44.019
C   56.594   24.870   44.160
C   58.796   22.767   45.803
C   58.734   24.977   42.610
C   57.983   24.540   41.363
N   53.775   25.715   44.254
C   54.766   26.648   44.278
C   54.156   27.984   44.457
C   52.786   27.745   44.422
C   52.535   26.355   44.246
C   54.843   29.302   44.605
C   51.465   28.321   44.532
O   51.246   29.507   44.788
C   50.336   27.236   44.228
C   49.801   27.566   42.903
O   50.388   27.804   41.842
O   48.430   27.658   42.995
C   47.801   28.208   41.804
C   47.806   21.966   38.402
C   48.768   21.640   37.322
C   48.829   20.692   36.411
C   47.747   19.563   36.453
C   50.016   20.536   35.513
C   51.076   19.571   35.927
C   51.255   18.524   34.794
C   52.647   17.881   34.879
C   53.651   18.786   34.079
C   52.510   16.465   34.422
C   52.122   16.319   32.980
C   53.145   15.498   32.110
C   52.443   14.332   31.399
C   52.013   13.258   32.405
C   53.400   13.730   30.290
C   53.008   14.033   28.855
C   54.156   14.715   28.101
C   54.991   13.651   27.258
C   56.375   14.218   26.945
C   54.169   13.331   25.962
H   50.579   20.263   44.225
H   57.040   20.650   44.289
H   56.972   26.827   44.118
H   48.641   24.705   44.507
H   48.888   21.986   43.680
H   48.661   21.432   45.920
H   47.909   23.107   45.840
H   49.568   22.857   46.468
H   48.389   25.400   42.290
H   47.823   23.705   42.536
H   50.456   23.535   41.572
H   49.735   24.679   40.472
H   52.227   17.626   43.514
H   51.297   18.472   44.661
H   52.706   17.520   45.276
H   57.490   17.226   44.193
H   57.230   18.602   45.148
H   57.340   18.823   43.389
H   58.923   22.739   43.602
H   58.690   25.204   44.718
H   58.003   22.755   46.550
H   59.490   23.535   46.148
H   59.295   21.798   45.774
H   59.732   24.585   42.408
H   58.805   26.062   42.538
H   57.193   23.801   41.497
H   58.602   24.211   40.528
H   57.460   25.416   40.980
H   55.525   29.325   43.755
H   55.355   29.444   45.556
H   54.193   30.156   44.415
H   49.578   27.364   45.001
H   47.796   27.471   41.001
H   48.394   29.036   41.416
H   46.762   28.474   41.999
H   46.885   22.330   37.946
H   47.509   21.112   39.010
H   49.692   22.216   37.369
H   47.496   19.302   35.425
H   46.790   19.766   36.934
H   48.189   18.678   36.909
H   50.498   21.512   35.459
H   49.688   20.438   34.479
H   50.870   18.995   36.829
H   51.984   20.155   36.081
H   51.139   18.993   33.817
H   50.524   17.717   34.858
H   53.057   17.771   35.883
H   54.361   18.182   33.514
H   54.307   19.305   34.778
H   53.163   19.412   33.331
H   51.727   16.043   35.052
H   53.454   15.927   34.513
H   51.979   17.284   32.495
H   51.188   15.764   32.886
H   54.042   15.142   32.617
H   53.493   16.150   31.308
H   51.545   14.715   30.913
H   51.019   13.308   32.849
H   52.687   13.376   33.254
H   52.071   12.243   32.012
H   53.530   12.663   30.471
H   54.332   14.197   30.608
H   52.224   14.784   28.757
H   52.674   13.165   28.286
H   54.819   15.376   28.659
H   53.693   15.471   27.467
H   55.066   12.843   27.986
H   56.729   15.121   27.443
H   56.552   14.367   25.880
H   56.983   13.328   27.109
H   53.668   12.375   26.106
H   54.780   13.092   25.092
H   53.454   14.108   25.693

