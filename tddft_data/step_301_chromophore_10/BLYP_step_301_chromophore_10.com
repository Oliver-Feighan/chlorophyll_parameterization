%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_301_chromophore_10 TDDFT with blyp functional

0 1
Mg   41.128   7.916   28.757
C   42.939   9.203   31.443
C   38.745   6.751   31.022
C   39.520   6.398   26.193
C   43.759   8.847   26.680
N   40.948   7.863   31.022
C   41.785   8.571   31.922
C   41.262   8.506   33.320
C   39.945   7.676   33.210
C   39.909   7.336   31.684
C   39.828   6.432   34.165
C   41.245   9.899   34.034
C   40.454   10.043   35.328
C   41.364   10.000   36.639
O   42.348   9.346   36.747
O   40.923   10.884   37.552
N   39.314   6.841   28.591
C   38.466   6.497   29.665
C   37.354   5.778   29.090
C   37.475   5.727   27.649
C   38.862   6.252   27.408
C   36.243   5.209   29.859
C   36.462   5.422   26.557
O   36.640   5.616   25.406
C   35.038   5.036   26.938
N   41.581   7.664   26.743
C   40.816   6.935   25.898
C   41.372   6.952   24.438
C   42.586   7.968   24.604
C   42.612   8.286   26.083
C   41.697   5.556   23.912
C   42.382   9.297   23.807
C   40.999   10.002   23.802
N   43.013   8.778   28.940
C   43.942   9.129   28.051
C   45.027   9.776   28.641
C   44.603   9.888   29.949
C   43.428   9.237   30.108
C   46.107   10.478   27.810
C   44.915   10.543   31.182
O   45.787   11.436   31.334
C   43.832   10.070   32.221
C   44.527   9.243   33.318
O   44.734   8.045   33.441
O   44.879   10.147   34.338
C   45.729   9.610   35.384
C   41.867   11.269   38.620
C   41.106   11.479   39.919
C   41.618   11.419   41.197
C   43.069   11.101   41.481
C   40.703   11.686   42.462
C   40.037   10.431   43.045
C   38.674   10.054   42.305
C   37.531   10.283   43.279
C   37.264   9.055   44.156
C   36.141   10.708   42.633
C   35.111   11.125   43.678
C   34.610   12.590   43.569
C   33.344   12.812   44.561
C   32.321   13.751   43.881
C   33.698   13.242   46.018
C   33.799   12.034   46.925
C   34.899   12.294   47.976
C   35.953   11.126   47.787
C   35.939   9.994   48.902
C   37.331   11.715   47.613
H   37.992   6.523   31.779
H   38.981   6.013   25.325
H   44.425   9.351   25.976
H   42.049   7.872   33.727
H   39.130   8.383   33.364
H   39.616   5.508   33.628
H   38.873   6.611   34.659
H   40.578   6.369   34.954
H   41.042   10.712   33.337
H   42.291   10.006   34.324
H   39.870   9.138   35.494
H   39.839   10.943   35.321
H   36.296   5.293   30.945
H   36.078   4.187   29.518
H   35.326   5.679   29.506
H   34.987   4.304   27.744
H   34.495   4.745   26.038
H   34.460   5.833   27.405
H   40.500   7.328   23.903
H   43.542   7.536   24.311
H   41.759   4.949   24.815
H   42.706   5.626   23.505
H   40.847   5.280   23.289
H   42.591   9.067   22.763
H   43.119   9.973   24.242
H   41.259   10.978   24.210
H   40.266   9.576   24.487
H   40.550   10.133   22.817
H   46.886   10.902   28.445
H   45.492   11.275   27.393
H   46.619   10.027   26.960
H   43.406   10.943   32.715
H   45.052   9.527   36.234
H   46.561   10.229   35.720
H   46.144   8.608   35.282
H   42.364   12.222   38.440
H   42.664   10.577   38.892
H   40.022   11.445   39.808
H   43.672   11.844   40.958
H   43.185   10.088   41.098
H   43.267   11.174   42.551
H   39.936   12.334   42.038
H   41.304   12.230   43.191
H   39.892   10.754   44.076
H   40.684   9.554   43.040
H   38.818   8.984   42.154
H   38.450   10.423   41.304
H   37.840   11.130   43.891
H   37.578   8.074   43.797
H   36.249   8.839   44.489
H   37.731   9.341   45.098
H   35.763   9.923   41.978
H   36.261   11.545   41.945
H   35.586   11.153   44.659
H   34.158   10.599   43.636
H   34.301   12.754   42.537
H   35.339   13.330   43.899
H   32.768   11.891   44.643
H   32.328   13.708   42.792
H   32.560   14.769   44.188
H   31.375   13.488   44.355
H   32.849   13.857   46.316
H   34.619   13.813   45.901
H   33.855   11.124   46.328
H   32.921   11.902   47.557
H   34.585   12.274   49.020
H   35.428   13.223   47.763
H   35.744   10.535   46.896
H   35.183   10.128   49.675
H   36.897   9.998   49.421
H   35.792   9.003   48.471
H   37.408   12.741   47.970
H   37.653   11.673   46.572
H   38.066   11.356   48.333

