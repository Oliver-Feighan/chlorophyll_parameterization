%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_501_chromophore_19 TDDFT with PBE1PBE functional

0 1
Mg   25.474   50.808   26.360
C   23.428   51.775   29.140
C   27.780   49.800   28.469
C   27.486   50.748   23.727
C   22.891   52.356   24.248
N   25.555   50.754   28.557
C   24.640   51.222   29.521
C   25.122   51.018   31.007
C   26.564   50.285   30.683
C   26.673   50.244   29.162
C   27.794   51.069   31.313
C   24.145   50.220   31.939
C   24.756   49.667   33.221
C   24.216   50.301   34.516
O   23.885   51.460   34.610
O   24.239   49.333   35.532
N   27.353   50.240   26.123
C   28.224   49.807   27.095
C   29.509   49.420   26.552
C   29.385   49.702   25.134
C   27.975   50.196   24.942
C   30.619   48.819   27.278
C   30.387   49.509   24.009
O   30.133   49.668   22.848
C   31.785   49.180   24.461
N   25.246   51.521   24.284
C   26.255   51.326   23.398
C   25.833   51.764   21.970
C   24.323   52.139   22.167
C   24.105   51.951   23.656
C   26.648   52.908   21.276
C   23.287   51.290   21.349
C   23.202   51.696   19.861
N   23.590   51.862   26.561
C   22.664   52.335   25.660
C   21.523   52.771   26.362
C   21.756   52.644   27.680
C   23.067   52.092   27.804
C   20.267   53.219   25.626
C   21.304   52.861   29.058
O   20.285   53.343   29.481
C   22.426   52.306   30.059
C   23.024   53.488   30.753
O   23.930   54.230   30.447
O   22.333   53.579   31.945
C   22.728   54.667   32.871
C   23.877   49.902   36.806
C   24.739   49.271   37.882
C   25.950   49.656   38.234
C   26.791   50.591   37.355
C   26.590   49.050   39.537
C   26.587   47.518   39.787
C   25.377   46.991   40.564
C   25.624   45.635   41.416
C   25.574   44.318   40.507
C   24.675   45.666   42.657
C   25.445   46.043   43.992
C   24.926   47.149   44.808
C   26.153   47.992   45.289
C   26.978   47.314   46.360
C   25.804   49.489   45.636
C   26.981   50.497   45.773
C   26.833   51.873   45.160
C   27.712   52.147   43.866
C   27.624   53.600   43.450
C   27.356   51.204   42.686
H   28.624   49.489   29.088
H   28.238   50.843   22.940
H   22.183   52.676   23.480
H   25.367   52.000   31.412
H   26.549   49.299   31.148
H   28.475   50.357   31.779
H   27.602   51.880   32.015
H   28.413   51.495   30.524
H   23.788   49.470   31.232
H   23.211   50.695   32.240
H   25.834   49.824   33.197
H   24.450   48.627   33.331
H   30.630   47.839   26.801
H   30.427   48.807   28.351
H   31.610   49.254   27.150
H   31.916   48.098   24.467
H   32.150   49.826   25.260
H   32.403   49.702   23.731
H   25.751   50.835   21.405
H   24.205   53.214   22.029
H   26.084   53.794   20.983
H   26.952   52.496   20.314
H   27.504   53.280   21.839
H   22.293   51.431   21.772
H   23.592   50.254   21.503
H   23.572   52.712   19.722
H   22.156   51.574   19.580
H   23.827   51.097   19.198
H   19.377   53.068   26.237
H   20.205   52.681   24.680
H   20.363   54.258   25.310
H   22.017   51.578   30.760
H   21.795   55.047   33.287
H   23.278   55.481   32.399
H   23.314   54.085   33.583
H   22.867   49.614   37.099
H   24.041   50.971   36.937
H   24.267   48.530   38.527
H   26.259   50.863   36.443
H   27.662   49.996   37.080
H   27.031   51.468   37.955
H   26.066   49.528   40.364
H   27.633   49.347   39.646
H   27.442   47.117   40.330
H   26.542   47.058   38.800
H   24.676   46.656   39.800
H   24.894   47.741   41.190
H   26.640   45.661   41.811
H   25.640   44.612   39.459
H   24.639   43.822   40.767
H   26.463   43.710   40.677
H   24.307   44.646   42.768
H   23.881   46.377   42.430
H   26.494   46.167   43.722
H   25.299   45.138   44.583
H   24.344   46.729   45.627
H   24.219   47.806   44.301
H   26.864   48.071   44.466
H   26.616   46.346   46.708
H   27.151   47.987   47.200
H   27.966   47.111   45.946
H   25.225   49.584   46.555
H   25.057   49.861   44.935
H   27.920   50.140   45.350
H   27.237   50.626   46.825
H   26.869   52.642   45.932
H   25.776   52.021   44.937
H   28.747   51.970   44.158
H   28.661   53.918   43.335
H   27.043   54.045   44.258
H   27.089   53.718   42.508
H   27.531   51.734   41.750
H   26.294   50.977   42.592
H   28.001   50.326   42.680
